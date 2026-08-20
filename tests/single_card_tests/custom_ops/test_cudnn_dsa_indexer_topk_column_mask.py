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

"""Regression tests for the input-column mask in ``_indexer_top_k_unfused``.

``_indexer_top_k_unfused`` (``paddlefleet.cudnn_ops.indexer
.csa_indexer_fwd_cudnn``) replaced the cuDNN ``indexer_top_k_wrapper``. Its
contract is that ``seq_lens`` is a **hard per-row limit**: only compressed
columns ``[0, seq_lens)`` may be selected. Enforcing that requires masking the
*input* columns before ``paddle.topk``; blanking only the *output* slots
``>= seq_lens`` afterwards is not equivalent, because ``paddle.topk`` scans the
whole row and a finite score sitting at ``col >= seq_lens`` wins a low slot that
never gets blanked.

Whether the two differ depends on where the caller's ``-inf`` boundary sits:

* pure causal (``valid_range=None``) -- ``indexer_forward_wrapper`` already
  writes ``-inf`` past each row's ratio-causal limit, which *is* ``seq_lens``.
  No finite score exists past the limit, so the column mask is a no-op.
* CSA document mask -- the per-document valid window ends exactly at the
  document-local ratio-causal limit, so again ``seq_lens`` coincides with the
  kernel's ``-inf`` boundary and the mask is a no-op. This holds on both the
  packed-global (dense) path, where ``shift_scores_to_local_window`` re-fills
  the tail with ``-inf`` anyway, and the THD/varlen path.
* hybrid MLA indexer -- ``MQALatentAttention._indexer_valid_range`` clamps the
  window end ``csa_window_size`` *before* the diagonal (the forced local window
  is served separately), so ``seq_lens`` is strictly narrower than the causal
  limit and ``[seq_lens, causal_len)`` still holds finite scores. On the
  packed-global path ``shift_scores_to_local_window`` happens to blank them; on
  the THD path nothing does, and those out-of-window columns get selected. That
  path is production for the MQA+DSA layers, whose ``doc_lens`` is non-``None``
  whenever the documents tile the sequence.

The tests pin all three states against an inlined ``legacy`` replica of the
pre-fix helper, with a two-sided verdict: no-op cases must match **bitwise**,
and the hybrid-MLA THD case must differ **and** the legacy replica must still
exhibit out-of-window selections. The latter is the anti-vacuity guard: if it
ever stops holding, the premise of this test has changed and the fix needs to be
re-evaluated rather than silently kept.

This module is self-contained: it depends only on ``paddle`` and ``paddlefleet``
and never reaches outside the PaddleFleet checkout.
"""

import contextlib
import unittest
from unittest.mock import patch

import paddle

import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as indexer_mod

_SKIP_CONDITION = (
    not paddle.is_compiled_with_cuda()
    or paddle.device.cuda.device_count() == 0
    or paddle.device.cuda.get_device_capability()[0] != 10
)
_SKIP_REASON = "cuDNN DSA indexer requires Blackwell GPU (SM10x)"


def _require_sm100(cls):
    return unittest.skipIf(_SKIP_CONDITION, _SKIP_REASON)(cls)


def setUpModule():
    """Select the GPU here rather than at import time.

    ``ci/single_card_test.sh`` only accepts exit code 0, and a pytest run that
    collects zero tests exits 5. So this module must never skip itself during
    import/collection -- per-class ``skipIf`` decorators handle the SM100
    requirement instead.
    """
    if not _SKIP_CONDITION:
        paddle.set_device("gpu")


# =========================================================================
# Inlined "legacy" helper -- the differential reference
# =========================================================================


def _legacy_indexer_top_k_unfused(
    input_values: paddle.Tensor,
    seq_lens: paddle.Tensor,
    top_k: int,
    return_val: bool = True,
):
    """Byte-faithful replica of ``_indexer_top_k_unfused`` as introduced by #1666.

    That original version blanked only the **output slots** ``>= seq_lens`` and
    left the input columns untouched, while its docstring already declared
    "Expects input_values be masked by seq_lens". It is reproduced here (rather
    than imported) on purpose: ``src/`` no longer contains it, and this test's
    whole job is to compare against it. Keeping it inline also means the
    reference cannot drift when ``src/`` is refactored -- the differential
    verdicts below stay anchored to the exact pre-fix semantics.
    """
    # Note: paddle.topk doesn't allow k greater than the axis size.
    k = min(top_k, input_values.shape[-1])
    topk_values, topk_indices = paddle.topk(input_values, k, axis=-1)
    topk_indices = topk_indices.astype("int32")

    if k < top_k:
        topk_values = paddle.nn.functional.pad(
            topk_values, (0, 0, 0, top_k - k), value=float("-inf")
        )
        topk_indices = paddle.nn.functional.pad(
            topk_indices, (0, 0, 0, top_k - k), value=-1
        )

    topk_indices = paddle.where(
        paddle.arange(top_k, dtype="int32")
        < seq_lens.reshape([-1, 1]).cast("int32"),
        topk_indices,
        paddle.full_like(topk_indices, -1),
    )

    return {
        "indices": topk_indices,
        "values": topk_values if return_val else None,
    }


# =========================================================================
# Harness
# =========================================================================

_HEADS = 64  # cuDNN IndexerForward accepts H_i in {32, 64}
_DIM = 128  # ... and requires D_i == 128


def _make_indexer_inputs(s, ratio, seed=2026):
    """``(index_q, index_k_comp, weights, sk)`` for a batch of 1."""
    paddle.seed(seed)
    sk = s // int(ratio)
    index_q = paddle.randn([1, s, _HEADS, _DIM]).astype("bfloat16")
    index_k_comp = paddle.randn([1, sk, _DIM]).astype("bfloat16")
    weights = paddle.randn([1, s, _HEADS]).astype("bfloat16")
    return index_q, index_k_comp, weights, sk


def _doc_layout(doc_lens, ratio):
    """Per-token document token-start and compressed-column-start, int32 ``[S]``."""
    ratio = int(ratio)
    tok_start, col_start = [], []
    off_tok = off_col = 0
    for n in doc_lens:
        tok_start += [off_tok] * n
        col_start += [off_col] * n
        off_tok += n
        off_col += n // ratio
    return (
        paddle.to_tensor(tok_start, dtype="int32"),
        paddle.to_tensor(col_start, dtype="int32"),
    )


def _build_valid_range(doc_lens, ratio, window=0):
    """``valid_range [1, S, 2]`` plus its two columns as ``[S]`` tensors.

    ``window == 0`` reproduces the CSA document mask: the window ends exactly at
    the document-local ratio-causal limit. ``window > 0`` reproduces the hybrid
    MLA clamp of ``MQALatentAttention._indexer_valid_range``, which subtracts the
    forced local window and therefore ends *before* the causal limit.
    """
    s = sum(doc_lens)
    tok_start, col_start = _doc_layout(doc_lens, ratio)
    pos = paddle.arange(s, dtype="int32")
    doc_causal = (pos - tok_start + 1) // int(ratio)
    valid_end = col_start + paddle.clip(doc_causal - int(window), min=0)
    valid_range = paddle.stack([col_start, valid_end], axis=-1).unsqueeze(0)
    return valid_range, col_start, valid_end


def _run_topk_fwd(
    index_q,
    index_k_comp,
    weights,
    ratio,
    topk,
    valid_range=None,
    doc_lens=None,
    helper=None,
):
    """``cudnn_indexer_topk_fwd`` with ``helper`` swapped in for the top-k stage."""
    ctx = (
        patch.object(indexer_mod, "_indexer_top_k_unfused", helper)
        if helper is not None
        else contextlib.nullcontext()
    )
    with ctx:
        return indexer_mod.cudnn_indexer_topk_fwd(
            index_q,
            index_k_comp,
            weights,
            ratio=int(ratio),
            topk_effective=int(topk),
            valid_range=valid_range,
            doc_lens=doc_lens,
            seq_offset=0,
            return_topk_scores=False,
        )


def _count_outside_window(indices, valid_start, valid_end):
    """``(below, above)`` counts of returned ids outside ``[start, end)``."""
    ids = indices[0]  # [S, topk]
    valid = ids >= 0
    below = paddle.logical_and(valid, ids < valid_start.reshape([-1, 1]))
    above = paddle.logical_and(valid, ids >= valid_end.reshape([-1, 1]))
    return (
        int(below.cast("int32").sum()),
        int(above.cast("int32").sum()),
    )


class _IndexAssertions:
    """Structural invariants every ``cudnn_indexer_topk_fwd`` output must satisfy."""

    def assert_index_domain(self, indices, lengths, sk, topk, tag=""):
        """Every id is ``-1`` or in ``[0, sk)``, and ``topk_length`` counts them."""
        self.assertEqual(
            list(indices.shape[-1:]), [topk], f"{tag}: unexpected topk width"
        )
        ids = indices.astype("int32")
        in_domain = paddle.logical_or(
            ids == -1, paddle.logical_and(ids >= 0, ids < sk)
        )
        self.assertTrue(
            bool(in_domain.all()),
            f"{tag}: found ids outside {{-1}} U [0, {sk})",
        )
        counted = (ids >= 0).cast("int32").sum(axis=-1).cast("int32")
        self.assertTrue(
            bool((counted == lengths.astype("int32")).all()),
            f"{tag}: topk_length disagrees with the number of non-(-1) slots",
        )

    def assert_inside_window(self, indices, valid_start, valid_end, tag=""):
        below, above = _count_outside_window(indices, valid_start, valid_end)
        self.assertEqual(
            (below, above),
            (0, 0),
            f"{tag}: {below} ids before valid_start, {above} at/after valid_end",
        )


# =========================================================================
# 1) Helper-level minimal counterexample (no GPU kernel involved)
# =========================================================================


class TestIndexerTopKColumnMaskUnit(unittest.TestCase):
    """``_indexer_top_k_unfused`` alone, on hand-built scores.

    The minimal input that separates the two implementations: put the *largest*
    scores at columns ``>= seq_lens``. Masking the input columns keeps them out;
    blanking only the output slots does not. Pure ``paddle.topk`` / ``where``, so
    this runs on CPU as well as GPU.
    """

    #                  col:   0     1     2      3      4
    VALUES = [
        [1.0, 2.0, 3.0, 100.0, 200.0],  # seq_lens=3 -> decoys at 3, 4
        [5.0, -1.0, 50.0, 60.0, 70.0],  # seq_lens=1 -> decoys at 1..4
        [9.0, 8.0, 7.0, 6.0, 5.0],  # seq_lens=5 -> no decoy at all
    ]
    SEQ_LENS = [3, 1, 5]

    def _inputs(self):
        return (
            paddle.to_tensor(self.VALUES, dtype="float32"),
            paddle.to_tensor(self.SEQ_LENS, dtype="int32"),
        )

    def test_decoys_beyond_seq_lens_are_not_selected(self):
        values, seq_lens = self._inputs()
        out = indexer_mod._indexer_top_k_unfused(
            values, seq_lens, top_k=2, return_val=False
        )
        self.assertEqual(
            out["indices"].tolist(),
            [[2, 1], [0, -1], [0, 1]],
            "column mask must confine top-k to [0, seq_lens)",
        )

    def test_legacy_replica_selects_the_decoys(self):
        """Anti-vacuity: the reference really does exhibit the bug."""
        values, seq_lens = self._inputs()
        out = _legacy_indexer_top_k_unfused(
            values, seq_lens, top_k=2, return_val=False
        )
        self.assertEqual(out["indices"].tolist(), [[4, 3], [4, -1], [0, 1]])

    def test_all_selected_columns_respect_seq_lens_for_any_top_k(self):
        """Also covers ``top_k > sk``, i.e. the ``-inf`` / ``-1`` pad branch."""
        values, seq_lens = self._inputs()
        sk = len(self.VALUES[0])
        for top_k in (1, 2, 4, sk, 7):
            out = indexer_mod._indexer_top_k_unfused(
                values, seq_lens, top_k=top_k, return_val=True
            )
            ids = out["indices"]
            self.assertEqual(list(ids.shape), [len(self.SEQ_LENS), top_k])
            for row, limit in enumerate(self.SEQ_LENS):
                picked = [int(c) for c in ids[row].tolist() if int(c) >= 0]
                with self.subTest(top_k=top_k, row=row):
                    self.assertTrue(
                        all(0 <= c < limit for c in picked),
                        f"row {row}: {picked} escapes [0, {limit})",
                    )
                    self.assertEqual(len(picked), min(limit, top_k, sk))
                    self.assertEqual(len(set(picked)), len(picked))

    def test_pure_causal_prefix_is_a_no_op(self):
        """When the row is already ``-inf`` past ``seq_lens``, both agree bitwise."""
        rows, sk = 6, 8
        paddle.seed(11)
        base = paddle.randn([rows, sk]).astype("float32")
        seq_lens = paddle.to_tensor([1, 2, 3, 4, 5, 8], dtype="int32")
        causal = paddle.where(
            paddle.arange(sk, dtype="int32") < seq_lens.reshape([-1, 1]),
            base,
            paddle.full_like(base, float("-inf")),
        )
        for top_k in (2, 4, 8):
            new = indexer_mod._indexer_top_k_unfused(
                causal, seq_lens, top_k=top_k, return_val=False
            )["indices"]
            old = _legacy_indexer_top_k_unfused(
                causal, seq_lens, top_k=top_k, return_val=False
            )["indices"]
            with self.subTest(top_k=top_k):
                self.assertEqual(new.tolist(), old.tolist())


# =========================================================================
# 2) State A -- pure causal: the column mask must be a no-op
# =========================================================================

_RATIOS = (1, 4, 128)
_SEQ_LENS_CASES = (256, 512)


def _doc_split(s, ratio):
    """First document length for the multi-doc layout, aligned to ``ratio``.

    Aligning keeps the compressed buffer document-tiled (``n // ratio`` columns
    each), which is what the THD path's ``cu_seqlens_k`` assumes; it also makes
    the second document's ``valid_range`` start non-zero so
    ``topk_local_to_global``'s remap is exercised.
    """
    split = ((40 + int(ratio) - 1) // int(ratio)) * int(ratio)
    return [split, s - split]


@_require_sm100
class TestPureCausalUnchanged(unittest.TestCase, _IndexAssertions):
    """``valid_range=None``: ``seq_lens`` == the kernel's own causal limit."""

    def test_bitwise_identical_to_legacy(self):
        for s in _SEQ_LENS_CASES:
            for ratio in _RATIOS:
                q, k, w, sk = _make_indexer_inputs(s, ratio)
                for topk in (32, 128):
                    topk = min(topk, sk)
                    tag = f"s={s} ratio={ratio} sk={sk} topk={topk}"
                    with self.subTest(s=s, ratio=ratio, topk=topk):
                        new_idx, new_len = _run_topk_fwd(q, k, w, ratio, topk)
                        old_idx, _ = _run_topk_fwd(
                            q,
                            k,
                            w,
                            ratio,
                            topk,
                            helper=_legacy_indexer_top_k_unfused,
                        )
                        self.assert_index_domain(
                            new_idx, new_len, sk, topk, tag
                        )
                        self.assertTrue(
                            bool((new_idx == old_idx).all()),
                            f"{tag}: column mask changed the pure-causal result",
                        )


# =========================================================================
# 3) State B -- CSA document mask: still a no-op, on both paths
# =========================================================================


@_require_sm100
class TestCsaDocMaskUnchanged(unittest.TestCase, _IndexAssertions):
    """``valid_range`` whose count equals the document-local causal count.

    Covers both back ends selected by ``doc_lens``: ``None`` -> packed-global
    dense path, non-``None`` -> cuDNN THD/varlen path.
    """

    def test_bitwise_identical_to_legacy(self):
        for s in _SEQ_LENS_CASES:
            for ratio in _RATIOS:
                q, k, w, sk = _make_indexer_inputs(s, ratio)
                for doc_lens in ([s], _doc_split(s, ratio)):
                    valid_range, v_start, v_end = _build_valid_range(
                        doc_lens, ratio, window=0
                    )
                    for topk in (32, 128):
                        topk = min(topk, sk)
                        for thd in (False, True):
                            arg = doc_lens if thd else None
                            tag = (
                                f"s={s} ratio={ratio} doc_lens={doc_lens} "
                                f"topk={topk} thd={thd}"
                            )
                            with self.subTest(
                                s=s,
                                ratio=ratio,
                                doc_lens=tuple(doc_lens),
                                topk=topk,
                                thd=thd,
                            ):
                                self._check(
                                    q,
                                    k,
                                    w,
                                    ratio,
                                    topk,
                                    valid_range,
                                    arg,
                                    sk,
                                    v_start,
                                    v_end,
                                    tag,
                                )

    def _check(
        self, q, k, w, ratio, topk, valid_range, arg, sk, v_start, v_end, tag
    ):
        new_idx, new_len = _run_topk_fwd(q, k, w, ratio, topk, valid_range, arg)
        old_idx, _ = _run_topk_fwd(
            q,
            k,
            w,
            ratio,
            topk,
            valid_range,
            arg,
            helper=_legacy_indexer_top_k_unfused,
        )
        self.assert_index_domain(new_idx, new_len, sk, topk, tag)
        self.assert_inside_window(new_idx, v_start, v_end, tag)
        # Legacy is inside the window here too -- that is the point of state B.
        self.assert_inside_window(old_idx, v_start, v_end, f"legacy {tag}")
        self.assertTrue(
            bool((new_idx == old_idx).all()),
            f"{tag}: column mask changed the CSA doc-mask result",
        )


# =========================================================================
# 4) State C -- hybrid MLA window clamp: the mask is load-bearing on THD
# =========================================================================

# ``(s, ratio, window, topk)``. ``window`` is in compressed-column units, as in
# ``_indexer_valid_range`` (production uses ratio 1, so tokens == columns; the
# compressed ratios are here to show the effect is not ratio-specific). Each
# entry is chosen so the excluded band ``[valid_end, causal_len)`` is non-empty
# for enough rows to be observable -- see ``test_legacy_leaks_out_of_window``.
_NARROW_CASES = (
    (256, 1, 128, 32),
    (256, 1, 128, 128),
    (512, 1, 128, 32),
    (512, 1, 128, 128),
    (256, 4, 16, 32),
    (512, 4, 16, 32),
    (512, 128, 1, 4),
)


@_require_sm100
class TestNarrowWindowNeedsColumnMask(unittest.TestCase, _IndexAssertions):
    """``valid_range`` end clamped ``window`` columns before the causal limit.

    This is ``MQALatentAttention._indexer_valid_range``: the forced local window
    is served by a separate index block, so the indexer must rank only what lies
    before it. The THD path leaves ``[valid_end, causal_len)`` finite, so the
    input-column mask is what keeps those columns out of the top-k.
    """

    def _cases(self):
        for s, ratio, window, topk in _NARROW_CASES:
            for doc_lens in ([s], _doc_split(s, ratio)):
                yield s, ratio, window, topk, doc_lens

    def test_dense_path_unchanged(self):
        """``doc_lens=None``: ``shift_scores_to_local_window`` already blanks the band."""
        for s, ratio, window, topk, doc_lens in self._cases():
            q, k, w, sk = _make_indexer_inputs(s, ratio)
            valid_range, v_start, v_end = _build_valid_range(
                doc_lens, ratio, window
            )
            tag = (
                f"s={s} ratio={ratio} window={window} "
                f"doc_lens={doc_lens} topk={topk} dense"
            )
            with self.subTest(
                s=s,
                ratio=ratio,
                window=window,
                topk=topk,
                doc_lens=tuple(doc_lens),
            ):
                new_idx, new_len = _run_topk_fwd(
                    q, k, w, ratio, topk, valid_range, None
                )
                old_idx, _ = _run_topk_fwd(
                    q,
                    k,
                    w,
                    ratio,
                    topk,
                    valid_range,
                    None,
                    helper=_legacy_indexer_top_k_unfused,
                )
                self.assert_index_domain(new_idx, new_len, sk, topk, tag)
                self.assert_inside_window(new_idx, v_start, v_end, tag)
                self.assert_inside_window(
                    old_idx, v_start, v_end, f"legacy {tag}"
                )
                self.assertTrue(
                    bool((new_idx == old_idx).all()),
                    f"{tag}: dense path is expected to be mask-insensitive",
                )

    def test_thd_path_confines_topk_to_the_window(self):
        """The fixed helper: zero out-of-window ids on the production path."""
        for s, ratio, window, topk, doc_lens in self._cases():
            q, k, w, sk = _make_indexer_inputs(s, ratio)
            valid_range, v_start, v_end = _build_valid_range(
                doc_lens, ratio, window
            )
            tag = (
                f"s={s} ratio={ratio} window={window} "
                f"doc_lens={doc_lens} topk={topk} thd"
            )
            with self.subTest(
                s=s,
                ratio=ratio,
                window=window,
                topk=topk,
                doc_lens=tuple(doc_lens),
            ):
                new_idx, new_len = _run_topk_fwd(
                    q, k, w, ratio, topk, valid_range, doc_lens
                )
                self.assert_index_domain(new_idx, new_len, sk, topk, tag)
                self.assert_inside_window(new_idx, v_start, v_end, tag)

    def test_legacy_leaks_out_of_window(self):
        """Two-sided verdict / anti-vacuity guard.

        On the THD path the legacy helper must (a) disagree with the current one
        and (b) still return ids at or past ``valid_end``. If this ever fails,
        the premise behind the input-column mask has changed -- an upstream fix,
        a different top-k back end, or a ``valid_range`` that no longer sits
        inside the causal limit -- and the fix must be re-evaluated instead of
        being kept on faith.
        """
        for s, ratio, window, topk, doc_lens in self._cases():
            q, k, w, sk = _make_indexer_inputs(s, ratio)
            valid_range, v_start, v_end = _build_valid_range(
                doc_lens, ratio, window
            )
            tag = (
                f"s={s} ratio={ratio} window={window} "
                f"doc_lens={doc_lens} topk={topk} thd"
            )
            with self.subTest(
                s=s,
                ratio=ratio,
                window=window,
                topk=topk,
                doc_lens=tuple(doc_lens),
            ):
                new_idx, _ = _run_topk_fwd(
                    q, k, w, ratio, topk, valid_range, doc_lens
                )
                old_idx, _ = _run_topk_fwd(
                    q,
                    k,
                    w,
                    ratio,
                    topk,
                    valid_range,
                    doc_lens,
                    helper=_legacy_indexer_top_k_unfused,
                )
                self.assertFalse(
                    bool((new_idx == old_idx).all()),
                    f"{tag}: legacy and current agree -- this case no longer "
                    "distinguishes the two implementations",
                )
                _, new_above = _count_outside_window(new_idx, v_start, v_end)
                _, old_above = _count_outside_window(old_idx, v_start, v_end)
                self.assertEqual(new_above, 0, f"{tag}: current helper leaked")
                self.assertGreater(
                    old_above,
                    0,
                    f"{tag}: legacy helper no longer leaks out-of-window ids",
                )


if __name__ == "__main__":
    unittest.main()
