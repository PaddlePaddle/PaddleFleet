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
**attention** spans the full per-document causal set -- handed to dense FA4 as
an ``O(s)`` flashmask row bound and never materialised
(``_forward_full_causal`` -> ``_dense_attn``) -- and the KL is scored over that
same full causal set (``MQALatentAttention._forward_warmup`` -> one
``paddlefleet.tilelang_ops.csa_indexer_topk_fwd`` call at
``topk_effective = s_global``). The phase-3 cuDNN top-k kernel runs zero times
on this path, which is exactly why the mask semantics and the loss reduction
need pinning here and not only on the phase-3 path
(``dsa_indexer_use_sparse_loss=True``), where
``test_hybrid_mla_doc_equivalence.py`` and ``test_mqa_latent_attention.py``
already cover them.

Column sets are read through ``hybrid_mla_utils._CAPTURED``: the sparse kernel's
table literally, and on the dense path the column set the row bound the kernel
received implies (``RecordingMQA._dense_attn``).

What is proven here, and nowhere else:

* ``TestWarmupFullCausalTable`` -- row ``i`` of that column set is *exactly*
  ``[doc_start[i], i]``, on the layouts below plus a
  layout with genuine pad rows.
* ``TestWarmupCrossDocumentIsolation`` -- zero cross-document leakage: a packed
  batch equals per-document runs, in both eval (the ``:495`` early exit) and
  train (the ``:600`` branch). Dense FA4 derives its accumulation order from the
  flashmask row bounds, so repacking is only reproducible to a few bf16 ULPs, not
  bitwise; each layout therefore also measures a deliberately non-isolated
  forward, which misses by ~60x that bound.
* ``TestWarmupPadRows`` -- rows with ``is_valid == False``. ``_row_end`` cannot
  produce them (it turns the trailing gap into one final valid document), so
  ``_pad_row_end`` below keeps the gap folded into the last document instead,
  which is what a real packed batch's padding tail looks like.
* ``TestWarmupIndexerLossPrecision`` -- the KL column set (the whole causal set,
  with no cuDNN top-k call anywhere), the row mask coming from ``input_ids`` (not
  from the document metadata), the reduction denominator, and the claim that the
  KL target is the head-summed *full-causal* attention distribution.
* ``TestWarmupGradHealth`` -- the five indexer parameters, the detached indexer
  inputs, and the attention-side ``dq`` / ``dkv`` / ``d_sink``.

Shape caveat: ``WINDOW + INDEX_TOPK == 256`` in ``hybrid_mla_utils``, so a
``seqlen=256`` fixture has a *saturated* sparse budget -- the phase-3 top-k
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
    V_HEAD_DIM,
    WINDOW,
    H,
    _assert_agrees_to_bf16_ulps,
    _assert_isolation_is_observable,
    _build_module,
    _check_index_invariants,
    _create_mqa_config,
    _doc_meta,
    _fa4_module_hooks,
    _full_causal_indices,
    _make_inputs,
    _pad_row_end,
    _row_end,
)

setUpModule, tearDownModule = _fa4_module_hooks()

_EPS = 1e-10  # mqa_latent_attention._EPS, the KL/renormalisation epsilon

# The four required layouts, as ``(doc_lens, seqlen)``. ``_row_end`` turns any
# trailing gap into one more *valid* document, so "with a trailing gap" is still
# a well-formed multi-document layout here; genuine pad rows need
# ``_pad_row_end`` and live in ``_PAD_LAYOUTS``.
#
# ``seqlen=256`` is a *saturated* budget: ``WINDOW + INDEX_TOPK == 128 + 128 ==
# 256``, so at that shape the phase-3 sparse table already covers every row's
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


def _segments(row_end, seqlen):
    """``[(start, length), ...]`` for every document, from the production
    deriver, so the "run this document alone" reference isolates exactly what
    the packed kernel is supposed to."""
    _, _, _, doc_lens, doc_starts = _derive_csa_doc_boundaries(row_end, seqlen)
    return list(zip(doc_starts.numpy().tolist(), doc_lens.numpy().tolist()))


def _warmup_module(loss_coeff=0.01, sink=None, sparse_loss=False):
    """A ``"mqa_dsa"`` module with the phase switch off *from construction*.

    ``sparse_loss=True`` builds the phase-3 module instead, used only as the
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
# ``ctx``) -- the PyLayer phase 2 now attaches its loss through, imported into
# ``mqa_latent_attention`` from ``csa_attention``. The spy below binds by
# position, so a reordered signature would silently hand it the wrong tensors
# instead of failing. Assert the order rather than trust it.
_LOSS_ARGS = [
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


def _positional(columns, values=None):
    """Scatter a column-layout table back into position space.

    The tilelang indexer returns ``[b, s, width]`` tables in **column layout**:
    slot ``j`` of row ``i`` refers to token ``columns[i, j]``, ordered by
    descending score, with ``-1`` in the unused slots. That is not position
    order, so anything compared against a positional reference has to be
    scattered first. ``values is None`` returns the boolean "row ``i`` scored
    column ``c``" table, which is what the old dense ``causal_mask > -1`` was.
    """
    b, s, width = columns.shape
    dtype = bool if values is None else values.dtype
    out = np.zeros([b, s, width], dtype=dtype)
    for batch in range(b):
        for row in range(s):
            cols = columns[batch, row]
            keep = cols >= 0
            live = cols[keep]
            assert live.size == 0 or int(live.max()) < width, (
                f"row {row}: column id {int(live.max())} outside the "
                f"{width}-wide position space"
            )
            out[batch, row, live] = (
                True if values is None else values[batch, row, keep]
            )
    return out


@contextlib.contextmanager
def _capture_loss_args():
    """Capture what the warmup KL is actually reduced over.

    One spy, on the ``TileLangCSAIndexerLossAutoScaler`` boundary phase 2 now
    attaches its loss at, because every observable is an argument of that single
    call: ``P`` (``topk_probs``, already softmaxed by the kernel), ``Q``
    (``target``, from ``_attn_target``), the column ids, the row mask, the
    denominator and the coefficient. These are literally the tensors the
    backward (upstream's tilelang ``csa_indexer_bwd``) differentiates, so
    anything asserted here is what the gradient sees.

    ``cap["probs"]`` / ``cap["target"]`` / ``cap["columns"]`` are in the
    kernel's column layout; ``cap["live"]`` and ``cap["dense_target"]`` are the
    position-space scatters for tests that compare against a positional
    reference. A sum over the last axis -- which is all the KL does -- is
    permutation invariant and may be taken on the column layout directly.

    ``cap`` stays empty if the module was not in the warmup phase -- which is
    itself the assertion that phase 3 does not reach this code. Phase 3
    attaches through the *same* PyLayer now, so the discriminator is the
    ``indexer_backend`` tag: ``"tilelang"`` is phase 2's full-candidate kernel,
    ``"cudnn"`` is phase 3's top-k one, and only the former is recorded.
    """
    real = mqa_mod.TileLangCSAIndexerLossAutoScaler
    actual = [
        name
        for name in inspect.signature(real.forward).parameters
        if name != "ctx"
    ]
    assert actual == _LOSS_ARGS, (
        "TileLangCSAIndexerLossAutoScaler.forward was reordered: "
        f"{actual} != {_LOSS_ARGS}; the positional spy below would "
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
            loss_coeff=1.0,
            indexer_backend="tilelang",
            num_rows_override=None,
            loss_mask=None,
        ):
            if indexer_backend == "tilelang":
                cap["columns"] = topk_indices.numpy().copy()
                cap["probs"] = topk_probs.astype("float32").numpy().copy()
                cap["target"] = target.astype("float32").numpy().copy()
                cap["width"] = int(topk_indices.shape[-1])
                # ``loss_mask`` / ``num_rows_override`` are ``None`` when no
                # ``input_ids`` reached the layer -- the same unmasked branch
                # ``csa_attention`` takes -- so record that rather than assuming
                # a synthesised all-ones mask.
                cap["mask"] = (
                    None
                    if loss_mask is None
                    else loss_mask.astype("float32").numpy().copy()
                )
                cap["num_rows"] = (
                    None
                    if num_rows_override is None
                    else float(num_rows_override)
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
        if cap:
            cap["live"] = _positional(cap["columns"])
            cap["dense_target"] = _positional(cap["columns"], cap["target"])


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
    """Row ``i`` of the warmup column set is exactly ``[doc_start[i], i]``.

    The set is ``indices = doc_start + offsets`` masked by
    ``(indices > positions) | ~is_valid``, a pure integer function of the
    document bounds -- no kernel, no float -- so it is assertable as an exact set
    equality rather than a bound. Kernel-free, hence not GPU gated: production
    hands the same set to FA4 as an ``O(s)`` row bound, and that the bound really
    decodes to this set is what the ``_CAPTURED``-based classes below check.
    """

    def _assert_exact(self, row_end, seqlen):
        doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(
            row_end, seqlen
        )
        table = _full_causal_indices(1, seqlen, doc_start, is_valid).numpy()
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
    """Zero cross-document leakage: a packed batch equals per-document runs.

    Measured on the warmup path in both modes -- eval takes the ``:495`` early
    exit (indexer projections skipped entirely) and train takes the ``:600``
    branch (indexer runs, for the loss only).

    The comparison is allowed a few bf16 ULPs rather than bit equality, because the
    dense FA4 backend accumulates in an order it derives from the flashmask row
    bounds and repacking changes those bounds
    (``hybrid_mla_utils._assert_agrees_to_bf16_ulps``). Each layout also measures a
    deliberately non-isolated forward, which misses by ~500x that bound, so the
    tolerance cannot be hiding a leak.
    """

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        self.module = _warmup_module()

    def _check_isolation(self, layout, seqlen, training):
        tensors, w_v = _leaves(seqlen)
        query, key, x, qr = tensors
        row_end = _row_end(layout, seqlen)
        DSAIndexerLossLoggingHelper.tracker.clear()
        packed = _fp32(
            _forward(self.module, tensors, row_end, w_v, training=training)
        )
        # Control: the same inputs with every document boundary removed. This is
        # what a mask that failed to isolate would compute.
        DSAIndexerLossLoggingHelper.tracker.clear()
        no_isolation = _fp32(
            _forward(
                self.module,
                tensors,
                _row_end([seqlen], seqlen),
                w_v,
                training=training,
            )
        )
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
            worst = _assert_agrees_to_bf16_ulps(
                self,
                packed[:, sl],
                piece,
                f"{layout} doc@{start} ({'train' if training else 'eval'})",
            )
            # Row 0's causal set is the same either way, so the control is
            # only informative from the second document on.
            if start > 0:
                _assert_isolation_is_observable(
                    self,
                    worst,
                    float(np.abs(no_isolation[:, sl] - piece).max()),
                    packed[:, sl],
                )

    def test_packed_equals_single_eval(self):
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                self._check_isolation(layout, seqlen, training=False)

    def test_packed_equals_single_train(self):
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                self._check_isolation(layout, seqlen, training=True)


@_GPU
class TestWarmupPadRows(unittest.TestCase):
    """Rows outside every document must produce nothing and receive nothing.

    The warmup's mask is the caller's ``O(s)`` row bound handed to dense FA4, and
    on a pad row that bound leaves the visible span empty, so the kernel
    softmaxes over nothing. ``RecordingMQA`` writes the bound out as the column
    set it denotes (``_row_end_column_table``), where a pad row shows up as an
    all-``-1`` row -- checked here alongside the numbers. Both the output and
    ``dq`` must be exactly zero: a non-zero output would inject padding into the
    residual stream, and a non-zero ``dq`` would train on it. Checked with the
    sink OFF and ON, because the sink is the one column that survives the
    masking -- it is value-less, so it must still contribute nothing.
    """

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
    """The warmup KL: column set, row mask, denominator, target.

    Every number below is read at the ``TileLangCSAIndexerLossAutoScaler``
    boundary, i.e. exactly what the backward differentiates, and cross-checked
    against an independent fp32 recomputation of the logged scalar.
    """

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

    def test_kl_column_set_is_the_attention_column_set_and_no_topk_runs(self):
        """One column set serves both consumers, and no cuDNN top-k is called.

        This replaces the old "two distinct tables" expectation: phase 2 used to
        ask the indexer for a *wider* top-k table
        (``max(index_topk, min(2048, s//128*128))``) to score the KL over, which
        at the production ``index_topk=2048`` silently degenerated into the very
        phase-3 table it was supposed to widen. Phase 2 now scores every causal
        column instead, so the assertion is an equality between the KL's live
        columns and the attention table -- checked element-wise per row,
        including the diagonal the old indexer candidate range structurally
        excluded (``_indexer_valid_range`` clamped at
        ``causal_len - window_size``).

        Two call counts pin *which* selector produced those columns: the cuDNN
        top-k kernel (phase 3's) is called zero times, and the tilelang indexer
        exactly once per step at ``topk_effective == s``, its documented
        full-candidate mode.
        """
        import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as fwd_mod
        import paddlefleet.tilelang_ops as tl_mod

        module = _warmup_module()
        inner = fwd_mod.cudnn_indexer_topk_fwd
        inner_tl = tl_mod.csa_indexer_topk_fwd
        topk_calls = []
        tl_widths = []

        def recording(*args, **kwargs):
            topk_calls.append(1)
            return inner(*args, **kwargs)

        def recording_tl(*args, **kwargs):
            tl_widths.append(int(kwargs["topk_effective"]))
            return inner_tl(*args, **kwargs)

        fwd_mod.cudnn_indexer_topk_fwd = recording
        tl_mod.csa_indexer_topk_fwd = recording_tl
        try:
            for seqlen in (16, 128, 256, 300, 384, 512):
                with self.subTest(seqlen=seqlen):
                    before = len(tl_widths)
                    _, cap, attn_table = self._step(module, seqlen, [seqlen])
                    # The KL table is exactly the causal span -- no rounding. The
                    # tilelang wrapper pads ``topk_effective`` up to its block
                    # internally and crops the result back
                    # (``csa_indexer_fwd.py:430-462``), so short and non
                    # -power-of-two lengths are served too; ``seqlen=16`` is below
                    # the block size on purpose.
                    self.assertEqual(cap["target"].shape[-1], seqlen)
                    self.assertEqual(cap["probs"].shape[-1], seqlen)
                    self.assertEqual(cap["width"], seqlen)
                    self.assertEqual(attn_table.shape[-1], seqlen)
                    # One tilelang call, over every candidate column.
                    self.assertEqual(tl_widths[before:], [seqlen])
                    # The KL's live columns are exactly attention's columns.
                    live = cap["live"][0]
                    for row in range(seqlen):
                        cols = attn_table[0, row]
                        self.assertEqual(
                            set(cols[cols >= 0].tolist()),
                            set(np.flatnonzero(live[row]).tolist()),
                            f"s={seqlen} row {row}: KL and attention disagree",
                        )
                    # ... the diagonal included, on every row.
                    self.assertTrue(
                        bool(live[np.arange(seqlen), np.arange(seqlen)].all())
                    )
        finally:
            fwd_mod.cudnn_indexer_topk_fwd = inner
            tl_mod.csa_indexer_topk_fwd = inner_tl
        self.assertEqual(topk_calls, [], "warmup called the cuDNN top-k kernel")

    def test_pad_tail_excluded_from_indexer_loss_warmup(self):
        """Warmup counterpart of
        ``test_hybrid_mla_doc_equivalence.TestIndexerLossPadMaskRequestW``.

        One document spans the whole buffer, so ``is_valid`` is all ``True`` and
        the document metadata cannot express the padding tail. Only
        ``input_ids != pad_token_id`` can, and it must drive both the KL sum and
        its denominator -- the ``num_rows_override`` the backward divides by
        (``TileLangCSAIndexerLossAutoScaler.forward``'s ``ctx.num_rows``).
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
        self.assertEqual(float(cap["mask"].sum()), float(real_tokens))
        self.assertEqual(cap["num_rows"], float(real_tokens))
        mask = cap["mask"].reshape(-1)
        self.assertTrue(bool((mask[:real_tokens] == 1).all()))
        self.assertTrue(bool((mask[real_tokens:] == 0).all()))

        kl_per_row = self._kl_per_row(cap)
        ref = (kl_per_row * cap["mask"]).sum() / real_tokens * cap["coeff"]
        self.assertLess(abs(logged - float(ref)) / abs(float(ref)), 1e-5)
        # Two discriminators, so this is not a tautology.
        # (1) the denominator: had ``B*Sq`` driven it, the scalar would be
        # 256/200 = 1.28x smaller. (The old form of this check compared the
        # masked mean against the *plain* mean of the same rows, which
        # discriminated only because the top-k KL was near-zero on the padding
        # tail. The full-causal KL is the same order of magnitude on every row --
        # measured 0.7% apart here -- so that comparison no longer separates
        # anything and the denominator has to be pinned directly.)
        wrong_denominator = float(
            (kl_per_row * cap["mask"]).sum() / seqlen * cap["coeff"]
        )
        self.assertAlmostEqual(
            float(ref) / wrong_denominator, seqlen / real_tokens, delta=1e-6
        )
        self.assertGreater(abs(wrong_denominator - logged) / abs(logged), 0.2)
        # (2) the mask is not a no-op: the rows it drops carry real KL mass.
        dropped = float((kl_per_row * (1.0 - cap["mask"])).sum())
        self.assertGreater(dropped, 0.0)
        # Single card: cp_size == 1, so the coefficient reaches the backward
        # unscaled. The ``/cp_size`` branch is a multi-card concern.
        self.assertEqual(cap["coeff"], module.indexer_loss_coeff)

    def test_no_input_ids_uses_the_plain_row_mean(self):
        """Without ``input_ids`` the reduction is ``kl.mean() * coeff``.

        ``_indexer_loss_mask`` returns ``(None, None)`` and ``_forward_warmup``
        passes that straight down, which is the same unmasked branch
        ``csa_attention._compute_fused_indexer_target`` takes: the backward then
        falls back to the kernel's own ``1/(B*Sq)``, and only the *logged* scalar
        carries the ``/cp_size`` CP correction. So what to assert here is that
        both reach the backward as ``None`` -- a synthesised all-ones mask would
        silently change which denominator the gradient uses.
        """
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

    def test_kl_target_is_normalised_and_zero_off_the_causal_set(self):
        """The KL target is L1-normalised per row and exactly zero on columns the
        per-document causal mask excludes.

        The old form of this test asserted zero on ``-1`` top-k slots and
        allowed whole rows with *no* candidate at all (the window clamp could
        empty a row). Neither exists now: the column set is the causal set, so
        every row has at least its own diagonal. A row shorter than the ``s``-wide
        table still comes back ``-1``-padded, and those dead slots are where the
        "excluded column" assertion now lands.
        """
        seqlen, layout = 256, [40, 216]
        module = _warmup_module()
        _, cap, _ = self._step(module, seqlen, layout)
        target = cap["target"][0]
        masked = cap["columns"][0] < 0
        self.assertTrue(masked.any(), "layout masks nothing")
        self.assertFalse(masked.all(), "layout masks everything")
        self.assertEqual(
            float(np.abs(target[masked]).max()),
            0.0,
            "a masked column carries probability mass",
        )
        self.assertGreaterEqual(float(target.min()), 0.0)
        self.assertLess(
            float(np.abs(target.sum(axis=-1) - 1.0).max()),
            1e-5,
            "target rows are not L1-normalised",
        )

    def test_kl_target_is_the_full_causal_attention_distribution(self):
        """The intended semantics, measured.

        In warmup both sides of the KL span the whole per-document causal set, so
        the target is the head-summed attention distribution over *all* causal
        columns, L1-normalised -- no restriction to an indexer candidate subset
        and no renormalisation onto it, which is what the previous revision did.

        The reference below is an independent fp32 numpy recomputation (full
        per-document causal softmax over all ``s`` columns, then head-summed and
        L1-normalised), so agreement is evidence about the semantics, not a
        restatement of the code. The residual is the bf16 rounding of the inputs.
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
        reference = head_sum / np.maximum(
            head_sum.sum(axis=-1, keepdims=True), _EPS
        )
        # The mask the implementation used must be the same predicate. The
        # kernel emits columns score-descending, so the comparison is on the
        # position-space scatter, not on the raw column order.
        np.testing.assert_array_equal(cap["live"][0], allowed)

        got = cap["dense_target"][0]
        max_abs = float(np.abs(got - reference).max())
        norm_rel = float(
            np.linalg.norm(got - reference) / np.linalg.norm(reference)
        )
        print(
            f"\n[warmup] KL target vs full-causal head sum: "
            f"max_abs={max_abs:.3e} norm_rel={norm_rel:.3e}"
        )
        self.assertLess(norm_rel, 3e-2)
        self.assertLess(max_abs, 3e-2)

    def test_window_length_sequence_still_trains_the_indexer_in_warmup(self):
        """At ``s == csa_window_size`` phase 2 now learns, phase 3 still cannot.

        Pre-existing and unchanged for phase 3: ``_indexer_valid_range`` clamps
        the candidate range at ``causal_len - window_size``, so at ``s == window``
        every row's range is empty, the KL is exactly 0 and the indexer learns
        nothing that step. The old warmup shared that clamp and logged the same
        0.0 -- a floor in the very phase where the indexer does all of its
        learning. ``_forward_warmup`` passes ``window=0`` to
        ``_indexer_valid_range`` now, so the candidate range is the whole causal
        span and the KL is strictly positive at that shape (measured 9.10e-05);
        the phase-3 control still logs 0.0, which is what makes this a property of
        the change rather than of the fixture.
        """
        seqlen = WINDOW
        sparse = _warmup_module(sparse_loss=True)
        logged_sparse, cap_sparse, _ = self._step(sparse, seqlen, [seqlen])
        self.assertEqual(logged_sparse, 0.0)
        self.assertEqual(cap_sparse, {}, "phase 3 reached the warmup KL")

        module = _warmup_module()
        logged, cap, _ = self._step(module, seqlen, [seqlen])
        self.assertGreater(logged, 0.0)
        self.assertEqual(cap["target"].shape[-1], seqlen)
        self.assertGreater(float(np.abs(cap["target"]).max()), 0.0)
        # ... and it keeps scaling with the sequence, as before.
        logged_2w, cap_2w, _ = self._step(module, 2 * WINDOW, [2 * WINDOW])
        self.assertGreater(logged_2w, 0.0)
        self.assertGreater(float(np.abs(cap_2w["target"]).max()), 0.0)


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

    * the forced window already spans the whole document, so the phase-3
      attention table still contains the complete per-document causal set --
      asserted as ``want - got == empty set``, not as a count;
    * consequently all three ``hybrid_mla_attention`` shapes (phase 3, warmup,
      ``mqa_full_causal``) must produce the **same** output on these layouts,
      which is a much stronger statement than "no crash" -- bitwise for the two
      that share a backend and a row bound, and to a few bf16 ULPs across the
      sparse/dense backend boundary (``_assert_agrees_to_bf16_ulps``);
    * an all-``-1`` candidate row means a softmax over an empty set, the classic
      NaN source, so output and every gradient are checked finite;
    * and no starved row may borrow a column from a neighbouring document.
    """

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

    def test_all_three_modes_agree_when_starved(self):
        for layout, seqlen, note in _STARVED_LAYOUTS:
            _, starved = self._starved_rows(_row_end(layout, seqlen), seqlen)
            if starved < seqlen:
                continue  # only fully starved layouts must collapse to equality
            with self.subTest(layout=layout, note=note):
                phase3, _ = self._run("mqa_dsa", True, seqlen, layout, False)
                warmup, _ = self._run("mqa_dsa", False, seqlen, layout, False)
                causal, _ = self._run("mqa", None, seqlen, layout, False)
                # phase 3 runs the sparse kernel and warmup runs dense FA4, so
                # this is a cross-backend comparison: the column sets coincide
                # here, the reduction order does not
                # (``_assert_agrees_to_bf16_ulps``).
                _assert_agrees_to_bf16_ulps(
                    self,
                    _fp32(phase3),
                    _fp32(warmup),
                    "phase 3 != warmup although the window covers everything",
                )
                # Both of these are dense FA4 with the *same* row bound, so here
                # bit equality is the right bar and it holds.
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
