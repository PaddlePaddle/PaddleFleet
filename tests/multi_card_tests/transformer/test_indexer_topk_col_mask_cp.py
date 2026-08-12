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

"""``_indexer_top_k_unfused``'s column mask, under context parallel.

``#1666`` replaced the cuDNN radix ``indexer_top_k_wrapper`` with the pure
Paddle ``_indexer_top_k_unfused`` (``csa_indexer_fwd_cudnn.py:125``). That
helper blanks *output slots* past ``seq_lens`` but hands ``paddle.topk`` the
whole row, which is only safe when the caller's valid window already coincides
with the kernel's ``-inf`` boundary. Two callers do not agree on that:

* CSA (every ``csa_compress_ratios`` other than ``-2``): its ``valid_range`` end
  *is* the ratio-causal limit (``csa_attention.get_valid_range``), and on the
  dense path ``shift_scores_to_local_window`` fills the tail with ``-inf``
  (``docmask_utils.py:137-182``), so nothing finite sits past ``seq_lens``.
* hybrid MLA latent MQA + DSA: ``_indexer_valid_range`` deliberately clamps the
  end ``csa_window_size`` *before* the diagonal, because the forced local window
  already covers those tokens (``mqa_latent_attention.py:398-405``). On the THD
  path the scores are the kernel's own document-causal ones, so columns in
  ``[seq_lens, causal_len)`` are finite and ``topk`` will return them --
  duplicating the window and, at the diagonal, leaking the query's own column.

Masking the columns before ``topk`` fixes the second caller. This file is the
multi-card evidence for both halves of the claim, on the real kernels rather
than on a helper-level probe:

* ``TestCSAColumnMaskNoOpCP``: the CSA cuDNN indexer under CP, causal and
  docmask/THD, phase-2 and phase-3 widths -- the mask changes **nothing**
  bitwise, because nothing finite lives past ``seq_lens``.
* ``TestHybridMLAColumnMaskLoadBearingCP``: the latent MQA + DSA layer under CP
  -- without the mask the top-k *does* return out-of-window columns, so the
  no-op above is not a vacuous property of the helper.

No CSA CP test reached this helper before: ``test_csa_attention_cp.py``'s
real-kernel classes all run ``csa_indexer_backend="unfused"``, and its one cuDNN
class monkeypatches ``cudnn_indexer_topk_fwd`` away entirely.

Every collective (the aggregation all-reduces, and everything inside the layers)
is issued unconditionally on all ranks; only assertions are rank-local.

Run (2 or 4 GPUs)::

    PYTHONPATH=./third_party/PaddleFleet/src:./third_party/PaddleFormers \
    python -m paddle.distributed.launch --devices 0,1 --nnodes 1 \
        --master 127.0.0.1:<port> \
        third_party/PaddleFleet/tests/multi_card_tests/transformer/\
test_indexer_topk_col_mask_cp.py
"""

import contextlib
import unittest

import paddle
import paddle.distributed as dist

# Siblings on ``sys.path`` thanks to ``paddle.distributed.launch <thisfile>``,
# same import style as ``test_mqa_dsa_warmup_cp`` reusing ``test_mqa_dsa_cp``.
import test_csa_attention_cp as C
import test_mqa_dsa_cp as H

import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as M
from paddlefleet.transformer.csa_attention import CSADocMaskMetadata

# Captured before any patching so the spy can call the production helper.
_PRODUCTION_TOP_K = M._indexer_top_k_unfused

RATIO = 4
SQ_GLOBAL = 128
# cuDNN IndexerForward only accepts H_i in {32, 64} and D_i == 128
# (``_check_cudnn_indexer_shape_support``), which the CSA fixture's 16/32 does
# not satisfy -- so this file sets its own indexer dims.
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
# Two documents whose boundary (72) is not on a CP split, so each rank's
# ``valid_range`` slice is genuinely rank-specific.
DOC_ROW_END = [72] * 72 + [SQ_GLOBAL] * (SQ_GLOBAL - 72)


def setUpModule():
    # One fleet init for both harnesses; ``test_csa_attention_cp``'s builders
    # only read its module globals, so mirroring them is enough.
    H.setUpModule()
    C.CP_SIZE, C.CP_RANK, C.CP_GROUP = H.CP_SIZE, H.CP_RANK, H.CP_GROUP


def _unmasked_top_k(input_values, seq_lens, top_k, return_val=True):
    """``_indexer_top_k_unfused`` exactly as ``#1666`` shipped it.

    Output slots past ``seq_lens`` are blanked, input columns are not: this is
    the "unrepaired" behaviour, kept on the test side so the production helper
    does not need a switch (and so a future edit to it cannot silently redefine
    what this file compares against).
    """
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
        paddle.arange(top_k, dtype="int32") < seq_lens[:, None],
        topk_indices,
        paddle.full_like(topk_indices, -1),
    )
    return {
        "indices": topk_indices,
        "values": topk_values if return_val else None,
    }


def _count_out_of_range(indices, seq_lens_col):
    """Selected columns at or past the row's valid limit (``-1`` slots excluded)."""
    return int(((indices >= 0) & (indices >= seq_lens_col)).sum())


class _Spy:
    """Runs both helper versions on every real call and records the deltas.

    The production result is what gets returned, so the layer under observation
    behaves exactly as in production; the replica is pure extra arithmetic with
    no collectives, so installing this cannot change the communication pattern.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, input_values, seq_lens, top_k, return_val=True):
        prod = _PRODUCTION_TOP_K(input_values, seq_lens, top_k, return_val)
        unmasked = _unmasked_top_k(input_values, seq_lens, top_k, return_val)
        lens = seq_lens.reshape([-1, 1]).cast("int32")
        cols = paddle.arange(input_values.shape[-1], dtype="int32")
        finite = paddle.isfinite(input_values.cast("float32"))
        self.calls.append(
            {
                "rows": int(input_values.shape[0]),
                "cols": int(input_values.shape[-1]),
                "top_k": int(top_k),
                # The precondition: scores the kernel left finite past the
                # caller's valid limit. 0 => the mask cannot change anything.
                "finite_beyond": int(((cols >= lens) & finite).sum()),
                "oor_prod": _count_out_of_range(prod["indices"], lens),
                "oor_unmasked": _count_out_of_range(unmasked["indices"], lens),
                "drift": int(
                    (
                        prod["indices"].cast("int64")
                        != unmasked["indices"].cast("int64")
                    ).sum()
                ),
            }
        )
        return prod

    def total(self, key):
        return sum(c[key] for c in self.calls)

    def group_total(self, key):
        """Sum ``key`` over every rank; called on all ranks (collective)."""
        t = paddle.to_tensor([self.total(key)], dtype="int64")
        dist.all_reduce(t, group=H.CP_GROUP)
        return int(t[0])


@contextlib.contextmanager
def _spying():
    spy = _Spy()
    M._indexer_top_k_unfused = spy
    try:
        yield spy
    finally:
        M._indexer_top_k_unfused = _PRODUCTION_TOP_K


def _build_cudnn_csa(sparse_loss, window_size=32, topk=8):
    """A CSA layer on the real cuDNN indexer backend, CP-enabled."""
    cfg = C._build_csa_config(
        compress_ratio=RATIO,
        hidden_size=256,
        head_dim=64,
        q_lora_rank=64,
        csa_window_size=window_size,
        dsa_index_topk=topk,
        dsa_indexer_loss_coeff=0.0,
    )
    cfg.csa_indexer_backend = "cudnn"
    cfg.csa_sparse_attn_backend = "unfused"
    cfg.dsa_indexer_use_sparse_loss = sparse_loss
    cfg.dsa_index_n_heads = INDEX_HEADS
    cfg.dsa_index_head_dim = INDEX_HEAD_DIM
    paddle.seed(2026)
    csa = C._build_csa(cfg, RATIO, 64)
    csa.cp_group = H.CP_GROUP
    csa.cp_size = H.CP_SIZE
    csa.cp_rank = H.CP_RANK
    csa.cp_enabled = True
    csa.eval()
    return csa


@unittest.skipUnless(
    C._cudnn_indexer_available(),
    "the cuDNN indexer requires the cuDNN frontend on Blackwell (SM100)",
)
class TestCSAColumnMaskNoOpCP(unittest.TestCase):
    """CSA's own ``valid_range`` ends exactly where the kernel's ``-inf`` starts."""

    def _run(self, docmask, sparse_loss):
        sq_local = SQ_GLOBAL // H.CP_SIZE
        s, e = H.CP_RANK * sq_local, (H.CP_RANK + 1) * sq_local
        csa = _build_cudnn_csa(sparse_loss)

        meta = None
        if docmask:
            row_end = paddle.to_tensor(DOC_ROW_END, dtype="int32").reshape(
                [1, 1, SQ_GLOBAL, 1]
            )
            meta = CSADocMaskMetadata.build(
                ratio=RATIO,
                batch_size=1,
                seqlen=SQ_GLOBAL,
                startend_row_indices=row_end,
            )

        paddle.seed(1000)
        q = paddle.randn([1, SQ_GLOBAL, 8, 64], dtype=C.DTYPE)
        k = paddle.randn([1, SQ_GLOBAL, 1, 64], dtype=C.DTYPE)
        x = paddle.randn([1, SQ_GLOBAL, 256], dtype=C.DTYPE)
        qr = paddle.randn([1, SQ_GLOBAL, 64], dtype=C.DTYPE)

        with _spying() as spy, paddle.no_grad():
            csa.forward(
                q[:, s:e],
                k[:, s:e],
                k[:, s:e],
                None,
                x=x[:, s:e],
                qr=qr[:, s:e],
                docmask_meta=meta,
            )
        return spy

    def _assert_no_op(self, spy, tag):
        # ``group_total`` is collective: call it on every rank, before any
        # rank-local assertion can raise.
        group_calls = spy.group_total("rows")
        group_beyond = spy.group_total("finite_beyond")
        group_drift = spy.group_total("drift")
        print(
            f"[colmask-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: "
            f"calls={len(spy.calls)} shapes={[(c['rows'], c['cols'], c['top_k']) for c in spy.calls]} "
            f"finite_beyond={spy.total('finite_beyond')} "
            f"drift={spy.total('drift')} "
            f"oor_prod={spy.total('oor_prod')} "
            f"oor_unmasked={spy.total('oor_unmasked')} "
            f"[group: rows={group_calls} finite_beyond={group_beyond} "
            f"drift={group_drift}]",
            flush=True,
        )
        self.assertGreater(
            len(spy.calls),
            0,
            f"{tag}: the cuDNN indexer never reached "
            "_indexer_top_k_unfused, so this test proves nothing",
        )
        self.assertEqual(
            spy.total("finite_beyond"),
            0,
            f"{tag}: the kernel left {spy.total('finite_beyond')} finite "
            "scores past CSA's valid limit -- CSA's valid_range no longer "
            "coincides with the causal boundary, so the column mask is not a "
            "no-op for CSA any more",
        )
        self.assertEqual(
            spy.total("drift"),
            0,
            f"{tag}: {spy.total('drift')} index slots differ from the "
            "unmasked helper; the column mask changed CSA's selection",
        )
        self.assertEqual(spy.total("oor_prod"), 0, f"{tag}: oor (production)")
        self.assertEqual(spy.total("oor_unmasked"), 0, f"{tag}: oor (unmasked)")

    def test_1_causal(self):
        """Single document (``valid_range`` from the kernel's causal bound)."""
        for sparse_loss in (False, True):
            with self.subTest(sparse_loss=sparse_loss):
                spy = self._run(docmask=False, sparse_loss=sparse_loss)
                self._assert_no_op(spy, f"csa/causal/sparse={sparse_loss}")

    def test_2_docmask_thd(self):
        """Two documents: the THD/varlen path, where scores stay kernel-masked."""
        for sparse_loss in (False, True):
            with self.subTest(sparse_loss=sparse_loss):
                spy = self._run(docmask=True, sparse_loss=sparse_loss)
                self._assert_no_op(spy, f"csa/docmask/sparse={sparse_loss}")


class TestHybridMLAColumnMaskLoadBearingCP(unittest.TestCase):
    """The same mask is what keeps latent MQA + DSA inside its window.

    ``H._STRADDLE`` tiles ``s_global`` exactly, so ``doc_lens`` is handed to the
    kernel and the THD path runs (``mqa_latent_attention.py:574-589``) -- the one
    layout where the scores past ``seq_lens`` are still finite.
    """

    def _run(self, sparse_loss, loss_coeff):
        with _spying() as spy:
            H.run_core_cp(
                "mqa_dsa",
                H._STRADDLE,
                loss_coeff=loss_coeff,
                with_input_ids=loss_coeff > 0,
                sparse_loss=sparse_loss,
            )
        return spy

    def _report(self, spy, tag):
        totals = {
            k: spy.group_total(k)
            for k in (
                "rows",
                "finite_beyond",
                "drift",
                "oor_prod",
                "oor_unmasked",
            )
        }
        print(
            f"[colmask-cp{H.CP_SIZE} rank{H.CP_RANK}] {tag}: "
            f"calls={len(spy.calls)} "
            f"shapes={[(c['rows'], c['cols'], c['top_k']) for c in spy.calls]} "
            f"finite_beyond={spy.total('finite_beyond')} "
            f"drift={spy.total('drift')} "
            f"oor_prod={spy.total('oor_prod')} "
            f"oor_unmasked={spy.total('oor_unmasked')} "
            f"[group: {totals}]",
            flush=True,
        )
        return totals

    @H.U._GPU
    def test_1_phase3_mask_is_load_bearing(self):
        """window + top-k: without the mask the top-k re-selects window columns."""
        spy = self._run(sparse_loss=True, loss_coeff=0.1)
        totals = self._report(spy, "mqa_dsa/phase3")
        self.assertGreater(
            totals["rows"], 0, "the DSA indexer never reached the helper"
        )
        self.assertGreater(
            totals["finite_beyond"],
            0,
            "no finite score sat past valid_range, so this layout cannot "
            "distinguish the two helpers and the no-op result above is vacuous",
        )
        self.assertGreater(
            totals["oor_unmasked"],
            0,
            "the unmasked helper selected no out-of-window column, so the "
            "production mask is not what is keeping them out",
        )
        self.assertEqual(
            totals["oor_prod"],
            0,
            f"production selected {totals['oor_prod']} columns at or past "
            "valid_range end -- the forced window would be duplicated",
        )
        self.assertGreater(
            totals["drift"], 0, "the two helpers returned identical tables"
        )

    @H.U._GPU
    def test_2_warmup_never_reaches_the_topk_helper(self):
        """Phase 2 never reaches this helper, so the column mask cannot apply.

        ``_forward_warmup`` (``mqa_latent_attention.py:453-585``) supervises the
        indexer over the **whole** per-document causal span: one *tilelang*
        ``csa_indexer_topk_fwd`` with ``topk_effective=s_global`` and
        ``window=0`` (``mqa_latent_attention.py:524-541``), imported directly
        rather than through ``csa_indexer_backend``. So the cuDNN helper this
        file spies on -- the one that carries the column mask -- is not called
        on either the attention or the loss side, and with no forced window
        there is no window duplication for a mask to prevent. Asserted rather
        than dropped because it is the sharpest statement of the phase-2
        contract, and because it bounds ``test_1``'s claim: the mask is
        load-bearing for phase 3 only.
        """
        spy = self._run(sparse_loss=False, loss_coeff=0.1)
        totals = self._report(spy, "mqa_dsa/warmup")
        self.assertEqual(
            totals["rows"],
            0,
            f"phase 2 issued {totals['rows']} cuDNN top-k row(s) across the CP "
            "group; the warmup KL is supposed to go through the tilelang "
            "full-candidate kernel, which carries no column mask",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
