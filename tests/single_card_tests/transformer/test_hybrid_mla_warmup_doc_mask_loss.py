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

"""Document masking and indexer-loss arithmetic of the hybrid-MLA WARMUP phase.

The warmup phase is ``hybrid_mla_attention="mqa_dsa"`` with
``dsa_indexer_use_sparse_loss=False``: the indexer is still being learned, so
**attention** consumes the full per-document causal table
(``MQALatentAttention._build_full_causal_indices`` ->
``_build_mqa_causal_topk_idxs_from_doc_bounds``) while the indexer's top-k feeds
a *wider* KL table. Two different tables, one switch --  which is exactly why
the mask semantics and the loss reduction need pinning on this path and not only
on the phase-3/4 path (``dsa_indexer_use_sparse_loss=True``), where
``test_hybrid_mla_doc_equivalence.py`` and ``test_mqa_latent_attention.py``
already cover them.

What is proven here, and nowhere else:

* ``TestWarmupFullCausalTable`` -- row ``i`` of the attention table is
  *exactly* the column set ``[doc_start[i], i]``, on the layouts below plus a
  layout with genuine pad rows.
* ``TestWarmupCrossDocumentIsolation`` -- zero cross-document leakage, in the
  strong form the builder's docstring claims ("bit-identical to running each
  document on its own"): measured ``maxabs == 0.0`` exactly, in both eval (the
  ``:495`` early exit) and train (the ``:600`` branch).
* ``TestWarmupPadRows`` -- rows with ``is_valid == False``. ``_row_end`` cannot
  produce them (it turns the trailing gap into one final valid document), so
  ``_pad_row_end`` below keeps the gap folded into the last document instead,
  which is what a real packed batch's padding tail looks like.
* ``TestWarmupIndexerLossPrecision`` -- the KL candidate width formula, the row
  mask coming from ``input_ids`` (not from the document metadata), the
  reduction denominator, and the ``_attn_target`` claim that the KL target is
  the *full-causal* attention distribution restricted to the loss candidates
  and renormalised.
* ``TestWarmupGradHealth`` -- the five indexer parameters, the detached indexer
  inputs, and the attention-side ``dq`` / ``dkv`` / ``d_sink``.

Shape caveat: ``WINDOW + INDEX_TOPK == 256`` in ``hybrid_mla_utils``, so a
``seqlen=256`` fixture has a *saturated* sparse budget -- the phase-3/4 top-k
table would already equal the full causal set there, and no assertion at that
shape can tell the two apart. ``_LAYOUTS`` therefore also carries ``seqlen=512``
layouts, which is the discriminating shape.

Shared fixtures come from ``hybrid_mla_utils``.

Run::

    R=<erniebot checkout>
    PYTHONPATH=$R/third_party/PaddleFleet/src:$R/third_party/PaddleFormers \\
        CUDA_VISIBLE_DEVICES=5 FLAGS_selected_gpus=0 \\
        python -m pytest <this file> -q -p no:randomly
"""

import contextlib
import inspect
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlefleet.transformer.mqa_latent_attention as mqa_mod
from paddlefleet.transformer.csa_attention import _derive_csa_doc_boundaries
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper

from .hybrid_mla_utils import (
    _CAPTURED,
    _GPU,
    INDEX_TOPK,
    V_HEAD_DIM,
    WINDOW,
    H,
    _build_module,
    _check_index_invariants,
    _create_mqa_config,
    _doc_meta,
    _make_inputs,
    _row_end,
)

_EPS = 1e-10  # mqa_latent_attention._EPS, the KL/renormalisation epsilon

# The four required layouts, as ``(doc_lens, seqlen)``. ``_row_end`` turns any
# trailing gap into one more *valid* document, so "with a trailing gap" is still
# a well-formed multi-document layout here; genuine pad rows need
# ``_pad_row_end`` and live in ``_PAD_LAYOUTS``.
#
# ``seqlen=256`` is a *saturated* budget: ``WINDOW + INDEX_TOPK == 128 + 128 ==
# 256``, so at that shape the phase-3/4 sparse table already covers every row's
# causal length and "attention takes the full causal set" is indistinguishable
# from "attention takes the indexer's top-k". The last two layouts are therefore
# at ``seqlen=512``, where the sparse budget covers at most 256 of a row's up to
# 512 causal columns -- only there does an exact-column-set assertion actually
# discriminate the warmup table from the top-k table.
_LAYOUTS = [
    ([256], 256),  # one document spanning the whole buffer
    ([40, 216], 256),  # two documents, tiles the buffer
    ([100, 50, 106], 256),  # three documents, none a multiple of the window
    ([127, 65], 256),  # trailing gap -> a third (valid) document of 64
    ([512], 512),  # discriminating: causal length 512 > window+topk = 256
    ([200, 312], 512),  # discriminating, multi-document
]

# Layouts whose trailing gap stays *outside* every document, i.e. real pad rows.
_PAD_LAYOUTS = [
    ([200], 256),  # 56 pad rows after a single document
    ([40, 88], 256),  # 128 pad rows after two documents
    ([100, 50, 60], 256),  # 46 pad rows after three documents
]


def _pad_row_end(doc_lens, seqlen):
    """``[1, 1, s, 1]`` int32 ``row_end`` that produces real pad rows.

    ``hybrid_mla_utils._row_end`` fills the trailing gap with ``seqlen``, which
    ``_derive_csa_doc_boundaries`` reads as one more document, so every row comes
    back ``is_valid``. Repeating the *last document's* end instead leaves
    ``doc_len_per_pos`` short of ``pos_in_doc`` for the tail, which is the
    ``is_valid == False`` state a packed batch's padding actually takes.
    """
    out = np.empty([seqlen], dtype="int32")
    pos, end = 0, 0
    for length in doc_lens:
        end = pos + length
        out[pos : min(end, seqlen)] = end
        pos = end
        if pos >= seqlen:
            break
    if pos < seqlen:
        out[pos:] = end
    return paddle.to_tensor(out).reshape([1, 1, seqlen, 1])


def _segments(row_end, seqlen):
    """``[(start, length), ...]`` for every document, from the production
    deriver, so the "run this document alone" reference isolates exactly what
    the packed kernel is supposed to."""
    _, _, _, doc_lens, doc_starts = _derive_csa_doc_boundaries(row_end, seqlen)
    return list(zip(doc_starts.numpy().tolist(), doc_lens.numpy().tolist()))


def _wide_loss_width(seqlen):
    """``mqa_latent_attention.py:562-566``: the warmup KL table width."""
    return max(INDEX_TOPK, min(2048, seqlen // 128 * 128))


def _warmup_module(loss_coeff=0.01, sink=None, sparse_loss=False):
    """A ``"mqa_dsa"`` module with the phase switch off *from construction*.

    ``sparse_loss=True`` builds the phase-3/4 module instead, used only as the
    control that decides whether a finding is specific to this change.
    """
    config = _create_mqa_config("mqa_dsa", loss_coeff=loss_coeff)
    config.dsa_indexer_use_sparse_loss = sparse_loss
    config.pad_token_id = 0
    module = _build_module(config, bf16=True, sink=sink)
    assert module.indexer is not None
    assert module.indexer_use_sparse_loss is sparse_loss
    return module


# Positional signature of ``TileLangCSAIndexerLossAutoScaler.forward`` (minus
# ``ctx``). The spy below binds by position, so a reordered upstream signature
# would silently hand it the wrong tensors -- which is exactly what happened
# when ``target`` moved from seventh to second: the spy then read
# ``index_k_comp`` as the top-k table (bf16 bits as ``uint16``, so ``< 0`` never
# fired) and the int32 column ids as the probabilities (``log`` of ``-1`` ->
# ``nan``). Assert the order instead of trusting it.
_LOSS_SCALER_ARGS = [
    "output",
    "target",
    "index_q",
    "weights",
    "index_k_comp",
    "topk_indices",
    "topk_probs",
    "loss_coeff",
    "indexer_backend",
    "num_rows_override",
    "loss_mask",
]


@contextlib.contextmanager
def _capture_loss_args():
    """Capture the arguments the layer hands ``TileLangCSAIndexerLossAutoScaler``.

    That is the only place the KL target, the top-k probabilities, the row mask
    and the reduction denominator are all visible at once, and it is the same
    boundary the backward consumes, so anything asserted here is what the
    gradient sees.
    """
    real = mqa_mod.TileLangCSAIndexerLossAutoScaler
    actual = [
        name
        for name in inspect.signature(real.forward).parameters
        if name != "ctx"
    ]
    assert actual == _LOSS_SCALER_ARGS, (
        "TileLangCSAIndexerLossAutoScaler.forward was reordered: "
        f"{actual} != {_LOSS_SCALER_ARGS}; the positional spy below would "
        "capture the wrong tensors"
    )
    cap = {}

    class _Spy:
        @staticmethod
        def apply(
            output,
            target,
            index_q,
            weights,
            index_k_comp,
            topk_indices,
            topk_probs,
            loss_coeff,
            indexer_backend="tilelang",
            num_rows_override=None,
            loss_mask=None,
        ):
            cap["indices"] = topk_indices.numpy().copy()
            cap["probs"] = topk_probs.astype("float32").numpy().copy()
            cap["target"] = target.astype("float32").numpy().copy()
            cap["mask"] = (
                None
                if loss_mask is None
                else loss_mask.astype("float32").numpy().copy()
            )
            cap["num_rows"] = (
                None if num_rows_override is None else float(num_rows_override)
            )
            cap["coeff"] = float(loss_coeff)
            return real.apply(
                output,
                target,
                index_q,
                weights,
                index_k_comp,
                topk_indices,
                topk_probs,
                loss_coeff,
                indexer_backend,
                num_rows_override,
                loss_mask,
            )

    mqa_mod.TileLangCSAIndexerLossAutoScaler = _Spy
    try:
        yield cap
    finally:
        mqa_mod.TileLangCSAIndexerLossAutoScaler = real


def _forward(module, tensors, row_end, w_v, training, input_ids=None):
    """One forward. ``tensors`` is ``(query, key, x, qr)``."""
    module.train() if training else module.eval()
    query, key, x, qr = tensors
    return module(
        query,
        key,
        None,
        None,
        row_end,
        v_b_proj_weight=w_v,
        x=x,
        qr=qr,
        input_ids=input_ids,
    )


def _leaves(seqlen, seed=1):
    """``(query, key, x, qr)`` as differentiable leaves, plus ``w_v``."""
    query, key, w_v, x, qr = _make_inputs(seqlen, seed=seed, with_hidden=True)
    tensors = [query, key, x, qr]
    for tensor in tensors:
        tensor.stop_gradient = False
    return tensors, w_v


def _fp32(tensor):
    """bf16 -> fp32 numpy; the widening is exact, so bit equality survives."""
    return tensor.cast("float32").numpy()


class TestWarmupFullCausalTable(unittest.TestCase):
    """Row ``i`` of the warmup attention table is exactly ``[doc_start[i], i]``.

    ``_build_full_causal_indices`` is ``indices = doc_start + offsets`` masked by
    ``(indices > positions) | ~is_valid``, a pure integer function of the
    document bounds -- no kernel, no float -- so it is assertable as an exact set
    equality rather than a bound. Kernel-free, hence not GPU gated.
    """

    def _assert_exact(self, row_end, seqlen):
        doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(
            row_end, seqlen
        )
        table = mqa_mod.MQALatentAttention._build_full_causal_indices(
            1, seqlen, doc_start, is_valid
        ).numpy()
        self.assertEqual(list(table.shape), [1, seqlen, seqlen])
        starts = doc_start.numpy()
        valid = is_valid.numpy().astype(bool)
        for row in range(seqlen):
            cols = table[0, row]
            got = set(cols[cols >= 0].tolist())
            want = (
                set(range(int(starts[row]), row + 1)) if valid[row] else set()
            )
            self.assertEqual(got, want, f"row {row}: column set is not exact")
            # The padding must be right-aligned ``-1``, never interleaved: the
            # kernel walks the row until its per-query length runs out.
            self.assertEqual(
                cols[: len(got)].tolist(),
                sorted(got),
                f"row {row}: selected columns are not a left-packed run",
            )
            self.assertTrue(
                bool((cols[len(got) :] == -1).all()),
                f"row {row}: non ``-1`` padding after the causal run",
            )
        _check_index_invariants(self, table, row_end, seqlen, expect_full=True)

    def test_exact_column_set_all_layouts(self):
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                self._assert_exact(_row_end(layout, seqlen), seqlen)

    def test_exact_column_set_with_pad_rows(self):
        for layout, seqlen in _PAD_LAYOUTS:
            with self.subTest(layout=layout, pad=True):
                row_end = _pad_row_end(layout, seqlen)
                _, is_valid = _doc_meta(row_end, seqlen)
                self.assertEqual(
                    int((~is_valid.astype(bool)).sum()),
                    seqlen - sum(layout),
                    "``_pad_row_end`` did not produce the pad rows",
                )
                self._assert_exact(row_end, seqlen)


@_GPU
class TestWarmupCrossDocumentIsolation(unittest.TestCase):
    """Zero cross-document leakage, in the builder's own strong form.

    ``_build_mqa_causal_topk_idxs_from_doc_bounds`` claims a packed batch is
    "bit-identical to running each document on its own". Measured on the warmup
    path: ``maxabs == 0.0`` exactly for every layout, in both modes -- eval takes
    the ``:495`` early exit (indexer projections skipped entirely) and train
    takes the ``:600`` branch (indexer runs, for the loss only). Exactness is
    expected rather than merely hoped for, because the sparse kernel reduces each
    query row over its own listed columns only, and those columns are identical
    in the packed and the single-document run once the column ids are shifted.
    """

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        self.module = _warmup_module()

    def _worst_leak(self, layout, seqlen, training):
        tensors, w_v = _leaves(seqlen)
        query, key, x, qr = tensors
        row_end = _row_end(layout, seqlen)
        DSAIndexerLossLoggingHelper.tracker.clear()
        packed = _fp32(
            _forward(self.module, tensors, row_end, w_v, training=training)
        )
        worst = 0.0
        for start, length in _segments(row_end, seqlen):
            sl = slice(start, start + length)
            DSAIndexerLossLoggingHelper.tracker.clear()
            piece = _fp32(
                _forward(
                    self.module,
                    (
                        query[:, sl].contiguous(),
                        key[:, sl].contiguous(),
                        x[:, sl].contiguous(),
                        qr[:, sl].contiguous(),
                    ),
                    _row_end([length], length),
                    w_v,
                    training=training,
                )
            )
            worst = max(worst, float(np.abs(packed[:, sl] - piece).max()))
        return worst

    def test_packed_equals_single_bitwise_eval(self):
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                worst = self._worst_leak(layout, seqlen, training=False)
                self.assertEqual(
                    worst, 0.0, f"{layout}: packed != single (eval)"
                )

    def test_packed_equals_single_bitwise_train(self):
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                worst = self._worst_leak(layout, seqlen, training=True)
                self.assertEqual(
                    worst, 0.0, f"{layout}: packed != single (train)"
                )


@_GPU
class TestWarmupPadRows(unittest.TestCase):
    """Rows outside every document must produce nothing and receive nothing.

    A pad row's table entry is all ``-1`` with a forced per-query length of 1
    (``_build_mqa_causal_topk_idxs_from_doc_bounds`` line 197-198), so the kernel
    softmaxes over an empty set. Both the output and ``dq`` must be exactly zero:
    a non-zero output would inject padding into the residual stream, and a
    non-zero ``dq`` would train on it. Checked with the sink OFF and ON, because
    the sink is the one column that survives the masking -- it is value-less, so
    it must still contribute nothing.
    """

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()

    def _check(self, layout, seqlen, sink):
        module = _warmup_module(sink=sink)
        row_end = _pad_row_end(layout, seqlen)
        _, is_valid = _doc_meta(row_end, seqlen)
        pad = ~is_valid.astype(bool)
        self.assertTrue(pad.any(), f"{layout}: no pad row was produced")

        tensors, w_v = _leaves(seqlen)
        _CAPTURED.clear()
        out = _forward(module, tensors, row_end, w_v, training=True)
        paddle.seed(7)
        upstream = paddle.randn([1, seqlen, H * V_HEAD_DIM]).cast("float32")
        (out.cast("float32") * upstream).sum().backward()

        table = _CAPTURED[-1]
        self.assertTrue(
            bool((table[0][pad] == -1).all()),
            f"{layout}: a pad row selected a column",
        )
        out_np = _fp32(out)[0]
        self.assertEqual(
            float(np.abs(out_np[pad]).max()),
            0.0,
            f"{layout}: pad row output is not exactly zero",
        )
        dq = _fp32(tensors[0].grad)[0]
        self.assertEqual(
            float(np.abs(dq[pad]).max()),
            0.0,
            f"{layout}: pad row dq is not exactly zero",
        )
        # The real rows must still be alive -- an all-zero output would satisfy
        # the assertions above for the wrong reason.
        self.assertGreater(float(np.abs(out_np[~pad]).max()), 0.0)
        self.assertGreater(float(np.abs(dq[~pad]).max()), 0.0)
        module.clear_gradients()

    def test_pad_rows_are_inert_sinkless(self):
        for layout, seqlen in _PAD_LAYOUTS:
            with self.subTest(layout=layout):
                self._check(layout, seqlen, sink=None)

    def test_pad_rows_are_inert_with_sink(self):
        sink = np.linspace(1.0, 3.0, H)
        for layout, seqlen in _PAD_LAYOUTS:
            with self.subTest(layout=layout):
                self._check(layout, seqlen, sink=sink)


@_GPU
class TestWarmupIndexerLossPrecision(unittest.TestCase):
    """The warmup KL: candidate width, row mask, denominator, target.

    Every number below is read at the ``TileLangCSAIndexerLossAutoScaler``
    boundary, i.e. exactly what the backward differentiates, and cross-checked
    against an independent fp32 recomputation of the logged scalar.
    """

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()

    def _step(self, module, seqlen, layout, input_ids=None, seed=1):
        """One training step; returns ``(logged_kl, captured, attn_table)``."""
        tensors, w_v = _leaves(seqlen, seed=seed)
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        with _capture_loss_args() as cap:
            out = _forward(
                module,
                tensors,
                _row_end(layout, seqlen),
                w_v,
                training=True,
                input_ids=input_ids,
            )
        out.cast("float32").sum().backward()
        logged = float(
            DSAIndexerLossLoggingHelper.tracker["values"]
            .astype("float32")
            .sum()
        )
        module.clear_gradients()
        return logged, cap, _CAPTURED[-1].copy()

    @staticmethod
    def _kl_per_row(cap):
        """fp32 recomputation of ``kl.sum(axis=-1)`` from the captured pair."""
        target, probs = cap["target"], cap["probs"]
        return (target * (np.log(target + _EPS) - np.log(probs + _EPS))).sum(
            axis=-1
        )

    def test_loss_width_formula_and_two_distinct_tables(self):
        """The KL table is ``max(index_topk, min(2048, s//128*128))`` wide while
        the attention table is ``s`` wide -- the deliberate double table.

        ``s=300`` is the case that separates them numerically (256 vs 300); the
        multiples of 128 make the two widths coincide as integers while still
        being different tables (full-causal vs indexer top-k), which the
        diagonal check below pins.
        """
        module = _warmup_module()
        for seqlen in (128, 256, 300, 384, 512):
            with self.subTest(seqlen=seqlen):
                _, cap, attn_table = self._step(module, seqlen, [seqlen])
                self.assertEqual(
                    cap["target"].shape[-1], _wide_loss_width(seqlen)
                )
                self.assertEqual(
                    cap["indices"].shape[-1], _wide_loss_width(seqlen)
                )
                self.assertEqual(attn_table.shape[-1], seqlen)
        # s=300: the widths genuinely differ, so a single table cannot be
        # serving both consumers.
        _, cap, attn_table = self._step(module, 300, [300])
        self.assertEqual(cap["target"].shape[-1], 256)
        self.assertEqual(attn_table.shape[-1], 300)
        # ... and the loss table structurally cannot be the attention one: the
        # indexer's candidate range is clamped to end ``window_size`` before the
        # query, so the diagonal is never in it while it always is in the
        # full-causal table.
        last = 299
        self.assertIn(last, set(attn_table[0, last].tolist()))
        self.assertNotIn(last, set(cap["indices"][0, last].tolist()))

    def test_pad_tail_excluded_from_indexer_loss_warmup(self):
        """Warmup counterpart of
        ``test_hybrid_mla_doc_equivalence.TestIndexerLossPadMaskRequestW``.

        One document spans the whole buffer, so ``is_valid`` is all ``True`` and
        the document metadata cannot express the padding tail. Only
        ``input_ids != pad_token_id`` can, and it must drive both the KL sum and
        its denominator -- the one the *backward* rescales the cuDNN kernel's
        built-in ``1/(B*Sq)`` into.
        """
        seqlen, real_tokens = 256, 200
        module = _warmup_module()
        ids = np.zeros([1, seqlen], dtype="int64")
        ids[0, :real_tokens] = np.arange(1, real_tokens + 1)
        logged, cap, _ = self._step(
            module, seqlen, [seqlen], input_ids=paddle.to_tensor(ids)
        )

        _, is_valid = _doc_meta(_row_end([seqlen], seqlen), seqlen)
        self.assertEqual(int(is_valid.astype("int32").sum()), seqlen)
        self.assertIsNotNone(cap["mask"])
        self.assertEqual(float(cap["mask"].sum()), float(real_tokens))
        self.assertEqual(cap["num_rows"], float(real_tokens))
        mask = cap["mask"].reshape(-1)
        self.assertTrue(bool((mask[:real_tokens] == 1).all()))
        self.assertTrue(bool((mask[real_tokens:] == 0).all()))

        kl_per_row = self._kl_per_row(cap)
        ref = (kl_per_row * cap["mask"]).sum() / real_tokens * cap["coeff"]
        self.assertLess(abs(logged - float(ref)) / abs(float(ref)), 1e-5)
        # Had ``is_valid`` driven the reduction, the loss would be measurably
        # different -- so this is a real discriminator, not a tautology.
        wrong = float(kl_per_row.mean() * cap["coeff"])
        self.assertGreater(abs(wrong - logged) / abs(logged), 0.1)
        # Single card: cp_size == 1, so the coefficient reaches the backward
        # unscaled. The ``/cp_size`` branch is a multi-card concern.
        self.assertEqual(cap["coeff"], module.indexer_loss_coeff)

    def test_no_input_ids_uses_the_plain_row_mean(self):
        """Without ``input_ids`` the reduction is ``kl.mean() * coeff`` and the
        backward keeps the kernel's own ``1/(B*Sq)`` (``num_rows_override`` is
        ``None``)."""
        seqlen = 256
        module = _warmup_module()
        logged, cap, _ = self._step(module, seqlen, [seqlen], input_ids=None)
        self.assertIsNone(cap["mask"])
        self.assertIsNone(cap["num_rows"])
        ref = float(self._kl_per_row(cap).mean() * cap["coeff"])
        self.assertLess(abs(logged - ref) / abs(ref), 1e-5)
        self.assertEqual(cap["coeff"], module.indexer_loss_coeff)
        self.assertEqual(
            module._indexer_loss_mask(None, 1, seqlen), (None, None)
        )

    def test_attn_target_is_normalised_and_zero_on_empty_slots(self):
        """``_attn_target`` rows are L1-normalised and exactly zero where the
        candidate slot is ``-1`` (including whole rows with no candidate at
        all)."""
        seqlen, layout = 256, [40, 216]
        module = _warmup_module()
        _, cap, _ = self._step(module, seqlen, layout)
        target, indices = cap["target"][0], cap["indices"][0]
        empty_slot = indices < 0
        empty_row = empty_slot.all(axis=-1)
        self.assertTrue(empty_row.any(), "layout has no empty candidate row")
        self.assertFalse(empty_row.all(), "layout has no live candidate row")
        self.assertEqual(
            float(np.abs(target[empty_slot]).max()),
            0.0,
            "a masked candidate slot carries probability mass",
        )
        self.assertGreaterEqual(float(target.min()), 0.0)
        rowsum = target.sum(axis=-1)
        self.assertEqual(float(np.abs(rowsum[empty_row]).max()), 0.0)
        self.assertLess(
            float(np.abs(rowsum[~empty_row] - 1.0).max()),
            1e-5,
            "live target rows are not L1-normalised",
        )

    def test_attn_target_is_the_renormalised_full_causal_distribution(self):
        """The intended semantics, measured.

        In warmup, attention consumes the **full per-document causal** set while
        the KL target is built over the *indexer's* candidate columns. So the
        target is the attention distribution restricted to those columns and
        renormalised -- deliberate, not a bug, and it is what makes the KL a
        ranking objective on the columns the indexer can actually move.

        The reference below is built the other way round from the
        implementation (full fp32 per-document causal softmax over all ``s``
        columns, head-summed, *then* restricted and renormalised) so agreement is
        evidence about the semantics, not a restatement of the code. The residual
        is the bf16 score rounding inside ``_attn_target``
        (``paddle.matmul`` on bf16 inputs, ~2^-8 relative), measured at
        norm-rel 9.68e-3 / maxabs 7.84e-3.
        """
        seqlen, layout = 256, [40, 216]
        module = _warmup_module()
        tensors, w_v = _leaves(seqlen)
        query, key = tensors[0], tensors[1]
        row_end = _row_end(layout, seqlen)
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        with _capture_loss_args() as cap:
            out = _forward(module, tensors, row_end, w_v, training=True)
        out.cast("float32").sum().backward()
        module.clear_gradients()

        doc_start, is_valid = _doc_meta(row_end, seqlen)
        positions = np.arange(seqlen)
        allowed = (
            (positions[None, :] <= positions[:, None])
            & (positions[None, :] >= doc_start[:, None])
            & is_valid.astype(bool)[:, None]
        )
        scores = (
            paddle.einsum(
                "shd,td->sht",
                query.detach()[0].cast("float32"),
                key.detach().squeeze(2)[0].cast("float32"),
            )
            * module.softmax_scale
        )
        scores = paddle.where(
            paddle.to_tensor(allowed).unsqueeze(1),
            scores,
            paddle.full_like(scores, -1e30),
        )
        head_sum = F.softmax(scores, axis=-1).sum(axis=1).numpy()

        indices = cap["indices"][0]
        reference = np.zeros_like(cap["target"][0])
        for row in range(seqlen):
            cols = indices[row]
            live = cols >= 0
            if not live.any():
                continue
            # Every indexer candidate must be inside the attention set, else the
            # renormalisation would not be well defined.
            self.assertTrue(
                bool(allowed[row][cols[live]].all()),
                f"row {row}: an indexer candidate is outside the causal set",
            )
            mass = head_sum[row, cols[live]]
            reference[row, live] = mass / max(mass.sum(), _EPS)

        got = cap["target"][0]
        max_abs = float(np.abs(got - reference).max())
        norm_rel = float(
            np.linalg.norm(got - reference) / np.linalg.norm(reference)
        )
        print(
            f"\n[warmup] _attn_target vs renormalised full-causal: "
            f"max_abs={max_abs:.3e} norm_rel={norm_rel:.3e}"
        )
        self.assertLess(norm_rel, 3e-2)
        self.assertLess(max_abs, 3e-2)

    def test_window_length_sequence_leaves_no_indexer_candidate(self):
        """PRE-EXISTING, not a warmup regression: at ``s == csa_window_size`` the
        indexer's candidate range is empty for every row
        (``_indexer_valid_range`` clamps at ``causal_len - window_size``), so the
        KL is exactly 0 and the indexer learns nothing that step.

        Recorded here because the warmup phase is where the indexer does all of
        its learning, so the floor matters. It is *not* introduced by the switch
        change: the control with ``dsa_indexer_use_sparse_loss=True`` logs the
        same 0.0, i.e. the cause is the window clamp shared by both phases.
        """
        seqlen = WINDOW
        for sparse_loss in (False, True):
            with self.subTest(sparse_loss=sparse_loss):
                module = _warmup_module(sparse_loss=sparse_loss)
                logged, cap, _ = self._step(module, seqlen, [seqlen])
                self.assertEqual(logged, 0.0)
                self.assertEqual(float(np.abs(cap["target"]).max()), 0.0)
                self.assertTrue(bool((cap["indices"] < 0).all()))
        # One token past the window and the indexer has candidates again.
        module = _warmup_module()
        logged, cap, _ = self._step(module, 2 * WINDOW, [2 * WINDOW])
        self.assertGreater(logged, 0.0)
        self.assertGreater(float(np.abs(cap["target"]).max()), 0.0)


@_GPU
class TestWarmupGradHealth(unittest.TestCase):
    """Warmup is where the indexer does all of its learning, so a silently
    gradient-free indexer parameter would waste the whole phase.

    Extends ``test_mqa_latent_attention.TestMQADSAWarmupPhase
    .test_indexer_gradients_flow_in_the_warmup_phase`` with the attention side
    (``dq`` / ``dkv`` / ``d_sink``) in the same step, on a multi-document layout
    and with the sink both OFF and ON. ``dkv`` is only checked finite and
    non-zero: it carries a known ~2e-3 run-to-run jitter from the atomic
    accumulation in ``csa_sparse_attn_bwd_cudnn``, shared with the ordinary
    CSA/HCA layouts, so its exact value is not a warmup property.
    """

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()

    def _assert_live(self, name, grad):
        self.assertIsNotNone(grad, f"{name} has no gradient")
        values = grad.cast("float32")
        self.assertTrue(
            bool(paddle.isfinite(values).all()),
            f"{name} gradient is not finite",
        )
        self.assertGreater(
            float(values.abs().max()), 0.0, f"{name} gradient is all zero"
        )

    def _check(self, sink):
        seqlen, layout = 256, [40, 216]
        module = _warmup_module(sink=sink)
        tensors, w_v = _leaves(seqlen)
        w_v.stop_gradient = False
        out = _forward(
            module, tensors, _row_end(layout, seqlen), w_v, training=True
        )
        paddle.seed(11)
        upstream = paddle.randn([1, seqlen, H * V_HEAD_DIM]).cast("float32")
        (out.cast("float32") * upstream).sum().backward()

        indexer = module.indexer
        for name, param in (
            ("indexer.wq_b.weight", indexer.wq_b.linear.weight),
            ("indexer.wk.weight", indexer.wk.linear.weight),
            ("indexer.k_norm.weight", indexer.k_norm.weight),
            ("indexer.k_norm.bias", indexer.k_norm.bias),
            ("indexer.weights_proj.weight", indexer.weights_proj.linear.weight),
        ):
            self._assert_live(name, param.grad)
        query, key, x, qr = tensors
        self._assert_live("dq", query.grad)
        self._assert_live("dkv", key.grad)
        self._assert_live("dw_v", w_v.grad)
        if sink is not None:
            self._assert_live("d_sink", module.softmax_offset.grad)
            self.assertEqual(
                module.softmax_offset.grad.dtype, module.softmax_offset.dtype
            )
        # The indexer learns from its own KL only: its inputs stay detached, so
        # no indexer gradient may leak into the backbone through ``x`` / ``qr``.
        self.assertIsNone(
            x.grad, "x.grad is not None: indexer input not detached"
        )
        self.assertIsNone(
            qr.grad, "qr.grad is not None: indexer input not detached"
        )
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)
        module.clear_gradients()

    def test_grad_health_sinkless(self):
        self._check(sink=None)

    def test_grad_health_with_sink(self):
        self._check(sink=np.linspace(1.0, 3.0, H))


# Layouts where the indexer has little or nothing left to choose from, because
# the forced window already covers each document. ``cand_i = max(causal_len_i -
# WINDOW, 0)``, so a document of length <= WINDOW gives *every* one of its rows
# an empty candidate range.
_STARVED_LAYOUTS = [
    ([64, 64, 64, 64], 256, "every doc half the window: 256/256 rows starved"),
    ([128, 128], 256, "every doc exactly the window: 256/256 rows starved"),
    ([1] * 8 + [120, 128], 256, "single-token docs mixed with window-sized"),
    ([2, 3, 5, 7, 11, 100], 128, "prime tiny docs, all far below the window"),
    ([129, 127], 256, "one row past the window: 255/256 starved"),
    ([1] * 8 + [248], 256, "single-token docs then one long document"),
    ([255, 1], 256, "long document plus a single-token tail"),
]


@_GPU
class TestStarvedIndexerCandidates(unittest.TestCase):
    """Packed documents too short for the indexer to pick anything.

    Real packing is dominated by short documents, so this is not a synthetic
    edge: ``_indexer_valid_range`` clamps the candidate end a full
    ``csa_window_size`` before the diagonal, so any document no longer than the
    window leaves *every* one of its rows with zero candidates. The interesting
    question is what that costs, and the answer must be "nothing":

    * the forced window already spans the whole document, so the phase-3/4
      attention table still contains the complete per-document causal set --
      asserted as ``want - got == empty set``, not as a count;
    * consequently all three ``hybrid_mla_attention`` shapes (phase 3/4, warmup,
      ``mqa_full_causal``) must produce **bit-identical** output on these
      layouts, which is a much stronger statement than "no crash";
    * an all-``-1`` candidate row means a softmax over an empty set, the classic
      NaN source, so output and every gradient are checked finite;
    * and no starved row may borrow a column from a neighbouring document.
    """

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def _module(self, mode, sparse_loss):
        config = _create_mqa_config(mode, loss_coeff=0.01)
        if sparse_loss is not None:
            config.dsa_indexer_use_sparse_loss = sparse_loss
        config.pad_token_id = 0
        return _build_module(config, bf16=True)

    def _run(self, mode, sparse_loss, seqlen, layout, backward):
        tensors, w_v = _leaves(seqlen)
        row_end = _row_end(layout, seqlen)
        module = self._module(mode, sparse_loss)
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        out = _forward(
            module,
            tensors,
            row_end,
            w_v,
            training=backward,
            input_ids=paddle.ones([1, seqlen], dtype="int64"),
        )
        grads = {}
        if backward:
            out.cast("float32").sum().backward()
            for name, tensor in zip(("query", "key", "x", "qr"), tensors):
                grads[name] = tensor.grad
            for name, param in module.named_parameters():
                grads[name] = param.grad
        return out, grads

    @staticmethod
    def _starved_rows(row_end, seqlen):
        doc_start, _, _, _, _ = _derive_csa_doc_boundaries(row_end, seqlen)
        starts = doc_start.numpy()
        causal_len = np.arange(seqlen) + 1 - starts
        return starts, int((np.maximum(causal_len - WINDOW, 0) == 0).sum())

    def test_starved_rows_keep_the_whole_causal_set(self):
        for layout, seqlen, note in _STARVED_LAYOUTS:
            with self.subTest(layout=layout, note=note):
                row_end = _row_end(layout, seqlen)
                starts, starved = self._starved_rows(row_end, seqlen)
                self.assertGreater(
                    starved, 0, "layout starves no row: the case is not covered"
                )
                self._run("mqa_dsa", True, seqlen, layout, backward=False)
                table = _CAPTURED[-1][0]
                for row in range(seqlen):
                    cols = table[row]
                    got = set(cols[cols >= 0].tolist())
                    want = set(range(int(starts[row]), row + 1))
                    self.assertEqual(
                        want - got,
                        set(),
                        f"row {row}: causal columns missing from the table",
                    )
                    self.assertEqual(
                        got - want,
                        set(),
                        f"row {row}: table has columns outside its document",
                    )

    def test_all_three_modes_agree_bitwise_when_starved(self):
        for layout, seqlen, note in _STARVED_LAYOUTS:
            _, starved = self._starved_rows(_row_end(layout, seqlen), seqlen)
            if starved < seqlen:
                continue  # only fully starved layouts must collapse to equality
            with self.subTest(layout=layout, note=note):
                phase3, _ = self._run("mqa_dsa", True, seqlen, layout, False)
                warmup, _ = self._run("mqa_dsa", False, seqlen, layout, False)
                causal, _ = self._run("mqa", None, seqlen, layout, False)
                self.assertEqual(
                    float(np.abs(_fp32(phase3) - _fp32(warmup)).max()),
                    0.0,
                    "phase 3/4 != warmup although the window covers everything",
                )
                self.assertEqual(
                    float(np.abs(_fp32(warmup) - _fp32(causal)).max()),
                    0.0,
                    "warmup != mqa_full_causal although the table is identical",
                )

    def test_starved_rows_stay_finite(self):
        for mode, sparse_loss in (
            ("mqa_dsa", False),
            ("mqa_dsa", True),
            ("mqa", None),
        ):
            for layout, seqlen, note in _STARVED_LAYOUTS:
                with self.subTest(mode=mode, sparse=sparse_loss, note=note):
                    out, grads = self._run(
                        mode, sparse_loss, seqlen, layout, backward=True
                    )
                    array = _fp32(out)
                    self.assertTrue(
                        np.isfinite(array).all(),
                        "an empty candidate row produced NaN/Inf output",
                    )
                    for name, grad in grads.items():
                        if grad is None:
                            continue
                        self.assertTrue(
                            np.isfinite(grad.cast("float32").numpy()).all(),
                            f"gradient {name} is not finite",
                        )


if __name__ == "__main__":
    unittest.main()
