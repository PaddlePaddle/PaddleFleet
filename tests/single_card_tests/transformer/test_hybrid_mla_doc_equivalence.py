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

"""Adversarial MASK / DOCUMENT-EQUIVALENCE validation for hybrid-MLA attention.

Validation agent A4. The gold standard proven here: N documents PACKED into a
single 8192-style sequence must equal running every document ALONE (same
weights, same content), elementwise on outputs AND gradients. The hybrid MLA
(``csa_compress_ratios == -2``) layers run one of two attentions, selected by
``hybrid_mla_attention`` and, within it, by whether the sublayers spec carries
an indexer:

* dense MHA (``hybrid_mla_attention="mha"``) -- the fp32 ``_dense_reference``
  below.
* the indexer-less :class:`MQALatentAttention` full-causal path -- latent MQA,
  mathematically equal to MHA (spec ``indexer=None``, which is what production
  builds for ``"mqa_full_causal"``).
* the :class:`MQALatentAttention` DSA path (``"mqa_dsa"``) -- forced window +
  top-k indexer.

Each is checked with the model-wide learnable per-head attention sink both ON
and OFF.

The shared direct-construction fixtures (geometry, stub sublayers, config
factory, module builder, fp32 dense reference) live in ``hybrid_mla_utils.py``,
a plain helper module rather than a test module, so the proof still does not
depend on another test's internals.

GPU-gated classes require the SM100+ FlashMLA sparse fwd + cuDNN DSA kernels;
the pure metadata / boundary-attack classes run on CPU.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.csa_attention import (
    _build_valid_range_from_doc_bounds,
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
    _validate_csa_docmask_shape,
)
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper

from .hybrid_mla_utils import (
    _CAPTURED,
    _GPU,
    DV,
    INDEX_TOPK,
    K_CHANNELS,
    V_HEAD_DIM,
    WINDOW,
    H,
    _assert_agrees_to_bf16_ulps,
    _assert_isolation_is_observable,
    _build_module,
    _check_index_invariants,
    _create_mqa_config,
    _dense_reference,
    _doc_meta,
    _fa4_module_hooks,
    _make_inputs,
    _row_end,
)

setUpModule, tearDownModule = _fa4_module_hooks()


def _doc_segments(layout, seqlen):
    """Derive ``(start, length)`` for every document (incl. the padding tail).

    Uses the production boundary deriver so the "single" runs mirror exactly
    what the packed kernel isolates.
    """
    row_end = _row_end(layout, seqlen)
    _, _, _, doc_lens, doc_starts = _derive_csa_doc_boundaries(row_end, seqlen)
    starts = doc_starts.numpy().tolist()
    lens = doc_lens.numpy().tolist()
    return list(zip(starts, lens))


def _relerr_maxmean(actual, expected, eps=1e-6):
    """Elementwise relative error stats: ``(norm_rel, max_abs, max_rel)``."""
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    denom = np.linalg.norm(e.reshape(-1))
    norm_rel = float(np.linalg.norm((a - e).reshape(-1)) / max(denom, 1e-12))
    absdiff = np.abs(a - e)
    max_abs = float(absdiff.max()) if absdiff.size else 0.0
    max_rel = (
        float((absdiff / (np.abs(e) + eps)).max()) if absdiff.size else 0.0
    )
    return norm_rel, max_abs, max_rel


# Adversarial layouts required by the task, as ``(layout, seqlen)`` pairs.
# ``seqlen`` >= sum(layout); any gap becomes a valid trailing document.
_LAYOUTS = [
    ([1, 1, 2], 4),  # all docs shorter than window
    ([127, 1, 128], 256),  # < window, single-token, == window
    ([128, 128], 256),  # two exact-window docs, tiles s
    ([40, 88, 128], 256),  # the classic packed case
    ([100, 50, 106], 256),  # lengths not divisible by 128
    ([127], 256),  # one sub-window doc + long tail
    ([128], 256),  # doc exactly the window + tail
    ([1, 1, 1, 1, 1, 1, 1, 1, 8], 256),  # >= 8 documents
    ([244, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 256),  # 1 long + many tiny
]


class TestMhaDenseMaskSemantics(unittest.TestCase):
    """Task 1 (``mha`` reference) + the packed==single gold standard proven on
    the exact fp32 dense math the ``mha`` path reduces to.

    ``mha`` uses the dense MLA attention (flash-mask document masking); the
    absorbed ``MQALatentAttention`` module is *not* on that path. Its
    mathematical content is ``_dense_reference``. Proving packed==single on the
    reference validates the mask semantics ``mha`` depends on, independent of
    any kernel; the ``mqa`` / ``mqa_dsa`` classes below then prove the kernels
    match this same reference bit-for-bit.
    """

    def _packed_vs_single(self, layout, seqlen, sink=None):
        query, key, w_v = _make_inputs(seqlen, seed=1)
        scale = float(K_CHANNELS**-0.5)
        packed = _dense_reference(
            query, key, w_v, _row_end(layout, seqlen), scale, sink=sink
        ).numpy()
        worst_abs, worst_rel = 0.0, 0.0
        for start, length in _doc_segments(layout, seqlen):
            q = query[:, start : start + length].contiguous()
            k = key[:, start : start + length].contiguous()
            piece = _dense_reference(
                q, k, w_v, _row_end([length], length), scale, sink=sink
            ).numpy()
            nr, max_abs, _ = _relerr_maxmean(
                packed[:, start : start + length], piece
            )
            worst_abs = max(worst_abs, max_abs)
            worst_rel = max(worst_rel, nr)
        return worst_abs, worst_rel

    def test_packed_equals_single_all_layouts_sinkless(self):
        # fp32 dense math: not bit-identical (softmax denom / ctx reductions run
        # over different lengths), but fp32 round-off only. Tight bound.
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                worst_abs, worst_rel = self._packed_vs_single(layout, seqlen)
                self.assertLess(worst_abs, 1e-4, f"{layout}: packed!=single")
                self.assertLess(worst_rel, 1e-5, f"{layout}: rel too large")

    def test_packed_equals_single_all_layouts_with_sink(self):
        sink = np.linspace(0.5, 2.5, H)
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                worst_abs, worst_rel = self._packed_vs_single(
                    layout, seqlen, sink=sink
                )
                self.assertLess(worst_abs, 1e-4, f"{layout}: packed!=single")
                self.assertLess(worst_rel, 1e-5, f"{layout}: rel too large")

    def test_sink_first_token_of_every_document(self):
        """Task 4: the sink is the *only* competitor for the first token of a
        document (its causal set is a single element), so a large sink logit
        must measurably drain that row's mass -- for every document, not just
        the first."""
        layout, seqlen = [40, 88, 128], 256
        query, key, w_v = _make_inputs(seqlen, seed=1)
        scale = float(K_CHANNELS**-0.5)
        big_sink = np.full(H, 20.0)  # exp(20) dominates a single real column
        with_sink = _dense_reference(
            query, key, w_v, _row_end(layout, seqlen), scale, sink=big_sink
        ).numpy()
        for start, _ in _doc_segments(layout, seqlen):
            row = np.abs(with_sink[0, start]).max()
            # A dominant sink drives the first-token output to ~0 (all mass to
            # the value-less sink column).
            self.assertLess(row, 1e-2, f"doc@{start}: sink absent on first row")


def _mask(vals, s):
    return paddle.to_tensor(np.asarray(vals, dtype="int32")).reshape(
        [1, 1, s, 1]
    )


class TestDeriveDocBoundariesAttack(unittest.TestCase):
    """Task 5: attack ``_derive_csa_doc_boundaries`` / doc_lens directly.

    Each test records whether a malformed input yields a clear error, a silent
    wrong result, or a negative doc_len. Findings feed the report's bug list.
    """

    def _derive(self, row_end, s):
        return _derive_csa_doc_boundaries(row_end, s)

    def test_single_full_document(self):
        ds, dl, iv, dlens, dstarts = self._derive(_mask([8] * 8, 8), 8)
        self.assertEqual(dstarts.numpy().tolist(), [0])
        self.assertEqual(dlens.numpy().tolist(), [8])
        self.assertTrue(bool(iv.all()))

    def test_all_zeros_degrades_to_invalid(self):
        ds, dl, iv, dlens, _ = self._derive(_mask([0] * 8, 8), 8)
        self.assertFalse(bool(iv.any()), "all-zero mask must select nothing")
        self.assertTrue(bool((dlens >= 0).all()), "negative doc_len")

    def test_all_ones_is_single_len1_doc(self):
        ds, dl, iv, dlens, _ = self._derive(_mask([1] * 8, 8), 8)
        self.assertEqual(int(iv.cast("int32").sum()), 1)
        self.assertEqual(dlens.numpy().tolist(), [1])

    def test_non_monotonic_marks_dip_invalid_no_corruption(self):
        # mask dips 4->2 at pos 2 then recovers.
        ds, dl, iv, dlens, dstarts = self._derive(
            _mask([4, 4, 2, 4, 8, 8, 8, 8], 8), 8
        )
        iv = iv.numpy().tolist()
        self.assertFalse(iv[2], "the non-monotonic dip row must be invalid")
        self.assertTrue(iv[0] and iv[1] and iv[3], "neighbours corrupted")
        self.assertTrue(bool((dlens >= 0).all()), "negative doc_len")

    def test_padding_only_tail(self):
        # doc of 5, then the buffer padded to s with row_end == s.
        ds, dl, iv, dlens, dstarts = self._derive(
            _mask([5, 5, 5, 5, 5, 8, 8, 8], 8), 8
        )
        self.assertEqual(dstarts.numpy().tolist(), [0, 5])
        self.assertEqual(dlens.numpy().tolist(), [5, 3])
        self.assertTrue(bool(iv.all()))

    def test_empty_trailing_document_is_not_negative(self):
        # last position claims to end at its own index -> zero-length tail.
        ds, dl, iv, dlens, _ = self._derive(
            _mask([3, 3, 3, 7, 7, 7, 7, 7], 8), 8
        )
        self.assertTrue(bool((dlens >= 0).all()), "negative doc_len produced")

    def test_seqlen_not_multiple_of_128(self):
        s = 130
        ds, dl, iv, dlens, _ = self._derive(_mask([130] * s, s), s)
        self.assertTrue(bool(iv.all()))
        self.assertEqual(dlens.numpy().tolist(), [130])

    def test_batch_gt_one_is_rejected_but_message_is_cryptic(self):
        """FINDING (A4-1): batch>1 raises a *cryptic* broadcast ValueError.

        ``_validate_csa_docmask_shape`` accepts ``[b,1,s,1]`` for b>1, but the
        deriver ``.flatten()``s to ``b*s`` then broadcasts against ``arange(s)``
        and dies with 'Broadcast dimension mismatch [s-1] vs [b*s-1]'. The code
        silently assumes a single packed (b==1) sequence. Production always
        packs to b==1 and ``MQALatentAttention`` rejects b!=1, so this is a
        hardening gap, not a live corruption.
        """
        s = 8
        b2 = paddle.to_tensor(
            np.array(
                [[3, 3, 3, 8, 8, 8, 8, 8], [5, 5, 5, 5, 5, 8, 8, 8]],
                dtype="int32",
            )
        ).reshape([2, 1, s, 1])
        # The shape validator does NOT catch it:
        _validate_csa_docmask_shape(b2, 2, s)
        with self.assertRaises(ValueError):
            self._derive(b2, s)

    def test_endpoint_beyond_seqlen_is_not_validated(self):
        """FINDING (A4-2): row_end > seqlen is accepted and yields doc_len > s.

        ``_validate_csa_docmask_shape`` checks shape only, never value range, so
        a mask claiming a document longer than the buffer produces
        ``doc_len == 10`` for ``s == 8``. Downstream index builders clamp by
        position so no crash results, but the metadata is silently wrong. The
        pack-time guards (``_pack_dsv4_logical_batch``) catch this in the real
        dataflow; the deriver itself does not.
        """
        ds, dl, iv, dlens, _ = self._derive(_mask([10] * 8, 8), 8)
        self.assertEqual(dlens.numpy().tolist(), [10])  # > seqlen, unvalidated

    def test_negative_endpoint_produces_negative_doc_len(self):
        """FINDING (W2, extends A4-2): a NEGATIVE row_end is accepted and yields
        a negative ``doc_len``.

        ``_validate_csa_docmask_shape`` checks shape only -- never dtype, upper
        bound, or lower bound -- so a mask of all ``-5`` passes shape validation
        and ``doc_len = mask - doc_start`` becomes ``-5``. It does not corrupt a
        real run (every row is then ``is_valid == False`` -> the MQA forward
        emits an all-zero output, no cross-doc leak, no crash), but the emitted
        metadata is silently wrong. A4's random sweep only covered well-formed
        monotonic masks, which never go negative; this is the adversarial
        counter-example. Production ``_pack_dsv4_logical_batch`` never emits a
        negative endpoint, so this is a LOW hardening gap, not a live bug.
        """
        ds, dl, iv, dlens, _ = self._derive(_mask([-5] * 8, 8), 8)
        self.assertFalse(
            bool(iv.any()), "negative-end mask must select nothing"
        )
        self.assertEqual(dlens.numpy().tolist(), [-5])  # negative, unvalidated

    def test_float_dtype_mask_is_silently_cast(self):
        """FINDING (W2): a float32 mask passes the shape validator and is
        silently ``cast("int64")`` by the deriver (truncation, not rejection).

        Integral float values behave like their int counterpart, but a
        non-integral endpoint would be silently truncated. Loud dtype rejection
        would be safer. LOW hardening gap; production always passes int32.
        """
        f = paddle.to_tensor(np.full([8], 8.0, dtype="float32")).reshape(
            [1, 1, 8, 1]
        )
        _validate_csa_docmask_shape(f, 1, 8)  # shape validator does NOT reject
        _, _, iv, dlens, _ = self._derive(f, 8)
        self.assertTrue(bool(iv.all()))
        self.assertEqual(dlens.numpy().tolist(), [8])  # float silently -> int

    def test_no_negative_doc_len_over_random_sweep(self):
        rng = np.random.default_rng(0)
        s = 64
        for _ in range(200):
            vals = np.sort(rng.integers(1, s + 1, size=s)).astype("int32")
            # make it a plausible monotonic non-decreasing end-row mask
            _, _, _, dlens, _ = self._derive(_mask(vals.tolist(), s), s)
            self.assertTrue(
                bool((dlens.numpy() >= 0).all()), f"neg doc_len for {vals}"
            )


class TestWindowIndexerPartitionMetadata(unittest.TestCase):
    """Task 2/3 at the metadata level (no kernel): the forced window and the
    indexer candidate range partition the per-document causal set exactly."""

    def setUp(self):
        self.module = _build_module(_create_mqa_config("mqa"))

    def test_window_indexer_partition_all_layouts(self):
        seqlen = 256
        for layout, _s in _LAYOUTS:
            with self.subTest(layout=layout):
                row_end = _row_end(layout, seqlen)
                doc_start, doc_len, is_valid, _, _ = _derive_csa_doc_boundaries(
                    row_end, seqlen
                )
                window = _build_window_topk_idxs_from_doc_bounds(
                    1, seqlen, WINDOW, doc_start, is_valid
                ).numpy()
                vr, row_empty = self.module._indexer_valid_range(
                    seqlen, doc_start, doc_len, is_valid
                )
                vr = vr.numpy()[0]
                row_empty = row_empty.numpy().reshape([seqlen])
                ds = doc_start.numpy()
                iv = is_valid.numpy()
                for q in range(seqlen):
                    win = {int(c) for c in window[0, q] if c >= 0}
                    cand = set(range(int(vr[q, 0]), int(vr[q, 1])))
                    if not iv[q]:
                        self.assertEqual(win, set())
                        self.assertEqual(cand, set())
                        self.assertTrue(bool(row_empty[q]))
                        continue
                    start = int(ds[q])
                    self.assertEqual(
                        win, set(range(max(start, q - WINDOW + 1), q + 1))
                    )
                    self.assertEqual(
                        win & cand, set(), "window/indexer overlap"
                    )
                    self.assertEqual(win | cand, set(range(start, q + 1)))


class TestHCAAgreesWithPackedMask(unittest.TestCase):
    """Task 7: the HCA (ratio-128) layers and the MQA layers must agree on the
    document boundaries of the same packed batch. They are both fed the output
    of a single ``_derive_csa_doc_boundaries`` call, so any disagreement means
    one builder mis-maps a document."""

    def _group_doc_id(self, doc_starts, doc_lens, ratio, n_groups):
        gid = np.full(n_groups, -1, dtype=np.int64)
        g = 0
        for d, (_st, ln) in enumerate(zip(doc_starts, doc_lens)):
            ng = ln // ratio
            gid[g : g + ng] = d
            g += ng
        return gid, g

    def test_ratio128_compressed_range_is_in_document(self):
        seqlen, ratio = 256, 128
        for layout, _s in _LAYOUTS:
            with self.subTest(layout=layout):
                row_end = _row_end(layout, seqlen)
                (doc_start, doc_len, is_valid, doc_lens, doc_starts) = (
                    _derive_csa_doc_boundaries(row_end, seqlen)
                )
                vr = _build_valid_range_from_doc_bounds(
                    ratio, seqlen, doc_start, doc_len, is_valid
                ).numpy()
                dstarts = doc_starts.numpy().tolist()
                dlens = doc_lens.numpy().tolist()
                n_total = sum(l // ratio for l in dlens)
                gid, used = self._group_doc_id(
                    dstarts, dlens, ratio, max(n_total, 1)
                )
                # doc index per position, from the SAME derive call MQA uses.
                pos_doc = (
                    np.searchsorted(
                        np.array(dstarts), doc_start.numpy(), side="right"
                    )
                    - 1
                )
                iv = is_valid.numpy()
                for q in range(seqlen):
                    lo, hi = int(vr[q, 0]), int(vr[q, 1])
                    if not iv[q] or hi <= lo:
                        continue
                    doc_ids = set(gid[lo:hi].tolist())
                    self.assertEqual(
                        doc_ids,
                        {int(pos_doc[q])},
                        f"{layout} row {q}: HCA compressed groups cross the "
                        "document boundary the MQA path uses",
                    )


class TestIndexerLossPadMaskRequestW(unittest.TestCase):
    """W2: the indexer-loss row mask comes from ``input_ids != pad_token_id``,
    NOT from the document metadata.

    A packed sequence's trailing padding is folded into the last document's row
    interval, so ``is_valid`` (derived from ``attn_mask_startend_row_indices``)
    still marks those pad rows valid. ``_indexer_loss_mask`` must therefore drop
    exactly the pad rows -- from both the KL sum and its denominator -- using
    ``input_ids``. No other test in this suite feeds ``input_ids`` to the layer.
    CPU-only: exercises the pure-tensor mask method, no kernel.
    """

    def test_pad_tail_excluded_from_indexer_loss(self):
        seqlen, real_tokens = 256, 200
        module = _build_module(_create_mqa_config("mqa"))
        # One document spanning the whole buffer -> is_valid is all True, i.e.
        # the mask cannot express the pad tail (the Request-W phenomenon).
        row_end = _row_end([seqlen], seqlen)
        _, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, seqlen)
        ids = np.zeros([1, seqlen], dtype="int64")
        ids[0, :real_tokens] = np.arange(1, real_tokens + 1)  # nonzero real ids
        loss_mask, valid_rows = module._indexer_loss_mask(
            paddle.to_tensor(ids), 1, seqlen
        )
        lm = loss_mask.numpy().reshape(-1)
        # is_valid over-counts by exactly the folded pad tail; input_ids is truth
        self.assertEqual(int(is_valid.cast("int32").sum()), seqlen)
        self.assertEqual(int(valid_rows), real_tokens)
        self.assertTrue(bool((lm[:real_tokens] == 1).all()))
        self.assertTrue(bool((lm[real_tokens:] == 0).all()))
        self.assertEqual(seqlen - int(valid_rows), seqlen - real_tokens)

    def test_no_input_ids_falls_back_to_plain_mean(self):
        module = _build_module(_create_mqa_config("mqa"))
        loss_mask, valid_rows = module._indexer_loss_mask(None, 1, 16)
        self.assertIsNone(loss_mask)
        self.assertIsNone(valid_rows)


def _leaf(t):
    x = t.clone().detach()
    x.stop_gradient = False
    return x


def _run_mqa(module, query, key, w_v, row_end, upstream, **extra):
    """Forward+backward one call; return (out_np, grads dict of np)."""
    q, k, wv = _leaf(query), _leaf(key), _leaf(w_v)
    module.clear_gradients()
    out = module(q, k, None, None, row_end, v_b_proj_weight=wv, **extra)
    (out.cast("float32") * upstream).sum().backward()
    grads = {
        "query": q.grad.cast("float32").numpy(),
        "key": k.grad.cast("float32").numpy(),
        "w_v": wv.grad.cast("float32").numpy(),
    }
    if module.softmax_offset is not None:
        grads["sink"] = module.softmax_offset.grad.cast("float32").numpy()
    return out.cast("float32").numpy(), grads


@_GPU
class TestMQAKernelPackedVsSingle(unittest.TestCase):
    """Task 1, ``mqa`` mode: the absorbed-MQA kernel packed run equals the
    per-document runs, on the output AND on every gradient, with the sink both
    OFF and ON. Also equals the fp32 dense reference."""

    def _module(self, sink=None):
        # bf16 throughout: the full-causal phase hands the sink to dense FA4 as
        # ``learnable_sink``, which asserts bf16 on it
        # (``flash_mask/cute/interface.py:598``), and that is what production's
        # ``build_softmax_offset`` gives via ``params_dtype``.
        return _build_module(_create_mqa_config("mqa"), bf16=True, sink=sink)

    def _fp32(self, tensor):
        return tensor.cast("float32").numpy()

    def _check_forward(self, sink, label):
        """Forward: packed vs single, to a few ULPs, + vs the fp32 dense reference.

        Not bit equality: dense FA4 derives its accumulation order from the
        flashmask row bounds, which repacking changes
        (``hybrid_mla_utils._assert_agrees_to_bf16_ulps``). Each multi-document
        layout also measures a deliberately non-isolated forward, so the
        tolerance is demonstrably too tight to hide cross-document attention.
        """
        module = self._module(sink=sink)
        for layout, seqlen in _LAYOUTS:
            with self.subTest(layout=layout):
                query, key, w_v = _make_inputs(seqlen, seed=1)
                row_end = _row_end(layout, seqlen)
                packed = self._fp32(
                    module(query, key, None, None, row_end, v_b_proj_weight=w_v)
                )
                # Control: the same inputs with every document boundary gone.
                no_isolation = self._fp32(
                    module(
                        query,
                        key,
                        None,
                        None,
                        _row_end([seqlen], seqlen),
                        v_b_proj_weight=w_v,
                    )
                )
                for start, length in _doc_segments(layout, seqlen):
                    sl = slice(start, start + length)
                    piece = self._fp32(
                        module(
                            query[:, sl].contiguous(),
                            key[:, sl].contiguous(),
                            None,
                            None,
                            _row_end([length], length),
                            v_b_proj_weight=w_v,
                        )
                    )
                    worst = _assert_agrees_to_bf16_ulps(
                        self,
                        packed[:, sl],
                        piece,
                        f"packed!=single (fwd, {label}) doc@{start}",
                    )
                    # Row 0's causal set is the same either way, so the
                    # control is informative from the second document on.
                    if start > 0:
                        _assert_isolation_is_observable(
                            self,
                            worst,
                            float(np.abs(no_isolation[:, sl] - piece).max()),
                            packed[:, sl],
                        )
                ref = _dense_reference(
                    query,
                    key,
                    w_v,
                    row_end,
                    module.softmax_scale,
                    sink=(None if sink is None else np.asarray(sink)),
                ).numpy()
                ref_rel, _, _ = _relerr_maxmean(packed, ref)
                self.assertLess(ref_rel, 5e-3, "kernel != dense reference")

    def test_forward_sinkless(self):
        self._check_forward(None, "sinkless")

    def test_forward_with_sink(self):
        self._check_forward(np.linspace(1.0, 3.0, H), "sink")

    def _bwd_check(self, sink):
        module = self._module(sink=sink)
        bwd_layouts = [
            ([40, 88, 128], 256),
            ([128, 128], 256),
            ([127, 1, 128], 256),
            ([1, 1, 1, 1, 1, 1, 1, 1, 8], 256),
            ([1, 1, 2], 4),
        ]
        worst = {"query": 0.0, "key": 0.0, "w_v": 0.0, "sink": 0.0}
        for layout, seqlen in bwd_layouts:
            query, key, w_v = _make_inputs(seqlen, seed=3)
            paddle.seed(7)
            upstream = paddle.randn([1, seqlen, H * V_HEAD_DIM]).cast("float32")
            _, g_packed = _run_mqa(
                module, query, key, w_v, _row_end(layout, seqlen), upstream
            )
            wv_accum = np.zeros_like(g_packed["w_v"])
            sink_accum = (
                np.zeros_like(g_packed["sink"]) if sink is not None else None
            )
            for start, length in _doc_segments(layout, seqlen):
                sl = slice(start, start + length)
                _, g_piece = _run_mqa(
                    module,
                    query[:, sl].contiguous(),
                    key[:, sl].contiguous(),
                    w_v,
                    _row_end([length], length),
                    upstream[:, sl].contiguous(),
                )
                for t in ("query", "key"):
                    nr, _, _ = _relerr_maxmean(g_packed[t][:, sl], g_piece[t])
                    worst[t] = max(worst[t], nr)
                wv_accum += g_piece["w_v"]
                if sink is not None:
                    sink_accum += g_piece["sink"]
            nr_wv, _, _ = _relerr_maxmean(g_packed["w_v"], wv_accum)
            worst["w_v"] = max(worst["w_v"], nr_wv)
            if sink is not None:
                nr_s, _, _ = _relerr_maxmean(g_packed["sink"], sink_accum)
                worst["sink"] = max(worst["sink"], nr_s)
        print(
            f"\n[A4] mqa backward norm-rel (sink={sink is not None}): {worst}"
        )
        return worst

    def test_backward_sinkless(self):
        worst = self._bwd_check(None)
        # cuDNN DSA backward reduces dq/dkv over the whole packed sequence with
        # a different grouping than the per-doc runs, so equivalence is exact in
        # real arithmetic but bf16 round-off only. Tight relative bound.
        self.assertLess(worst["query"], 2e-2, f"dquery packed!=single {worst}")
        self.assertLess(worst["key"], 2e-2, f"dkey packed!=single {worst}")
        self.assertLess(worst["w_v"], 2e-2, f"dw_v packed!=sum {worst}")

    def test_backward_with_sink(self):
        worst = self._bwd_check(np.linspace(1.0, 3.0, H))
        self.assertLess(worst["query"], 2e-2, f"dquery packed!=single {worst}")
        self.assertLess(worst["key"], 2e-2, f"dkey packed!=single {worst}")
        self.assertLess(worst["w_v"], 2e-2, f"dw_v packed!=sum {worst}")
        # The sink is a single [H] parameter shared by every row of every
        # document; its packed gradient must equal the sum over documents.
        self.assertLess(worst["sink"], 2e-2, f"dsink packed!=sum {worst}")


def _dsa_inputs(seqlen, seed=0):
    return _make_inputs(seqlen, seed=seed, with_hidden=True)


@_GPU
class TestMQADSAKernelPackedVsSingle(unittest.TestCase):
    """Task 1 (``mqa_dsa``) + Task 3. With a SATURATED budget the selected set
    is provably the full per-document causal set, so ``mqa_dsa`` must equal both
    the dense reference and the per-document runs exactly (up to bf16); sink
    ON/OFF. Genuinely-sparse soundness is covered by the capture class below.
    """

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()

    # window(128)+topk(128)=256 covers every causal length at s=256.
    _SAT = [
        ([40, 88, 128], 256),
        ([128, 128], 256),
        ([127, 1, 128], 256),
        ([100, 50, 106], 256),
        ([1, 1, 1, 1, 1, 1, 1, 1, 8], 256),
        ([244, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 256),
    ]

    def _module(self, sink=None):
        return _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.0), bf16=True, sink=sink
        )

    def test_forward_saturated_equals_reference_and_single(self, sink=None):
        module = self._module(sink=sink)
        module.eval()
        for layout, seqlen in self._SAT:
            with self.subTest(layout=layout, sink=sink is not None):
                query, key, w_v, x, qr = _dsa_inputs(seqlen, seed=2)
                row_end = _row_end(layout, seqlen)
                _CAPTURED.clear()
                packed = (
                    module(
                        query,
                        key,
                        None,
                        None,
                        row_end,
                        v_b_proj_weight=w_v,
                        x=x,
                        qr=qr,
                    )
                    .cast("float32")
                    .numpy()
                )
                # selected set is the full causal set (task 3).
                _check_index_invariants(
                    self, _CAPTURED[-1], row_end, seqlen, expect_full=True
                )
                ref = _dense_reference(
                    query,
                    key,
                    w_v,
                    row_end,
                    module.softmax_scale,
                    sink=(None if sink is None else np.asarray(sink)),
                ).numpy()
                ref_rel, _, _ = _relerr_maxmean(packed, ref)
                self.assertLess(ref_rel, 5e-3, "dsa != dense (saturated)")
                # packed vs per-document single runs.
                worst = 0.0
                for start, length in _doc_segments(layout, seqlen):
                    sl = slice(start, start + length)
                    piece = (
                        module(
                            query[:, sl].contiguous(),
                            key[:, sl].contiguous(),
                            None,
                            None,
                            _row_end([length], length),
                            v_b_proj_weight=w_v,
                            x=x[:, sl].contiguous(),
                            qr=qr[:, sl].contiguous(),
                        )
                        .cast("float32")
                        .numpy()
                    )
                    nr, _, _ = _relerr_maxmean(packed[:, sl], piece)
                    worst = max(worst, nr)
                self.assertLess(worst, 5e-3, f"{layout}: packed!=single (dsa)")

    def test_forward_saturated_with_sink(self):
        self.test_forward_saturated_equals_reference_and_single(
            sink=np.linspace(1.0, 3.0, H)
        )

    def test_backward_saturated_packed_vs_single(self):
        module = self._module(sink=np.linspace(1.0, 3.0, H))
        module.train()
        worst = {"query": 0.0, "key": 0.0, "w_v": 0.0, "sink": 0.0}
        for layout, seqlen in [([40, 88, 128], 256), ([128, 128], 256)]:
            query, key, w_v, x, qr = _dsa_inputs(seqlen, seed=5)
            paddle.seed(9)
            upstream = paddle.randn([1, seqlen, H * V_HEAD_DIM]).cast("float32")
            _, gp = _run_mqa(
                module,
                query,
                key,
                w_v,
                _row_end(layout, seqlen),
                upstream,
                x=x,
                qr=qr,
            )
            wv_accum = np.zeros_like(gp["w_v"])
            sink_accum = np.zeros_like(gp["sink"])
            for start, length in _doc_segments(layout, seqlen):
                sl = slice(start, start + length)
                _, gpi = _run_mqa(
                    module,
                    query[:, sl].contiguous(),
                    key[:, sl].contiguous(),
                    w_v,
                    _row_end([length], length),
                    upstream[:, sl].contiguous(),
                    x=x[:, sl].contiguous(),
                    qr=qr[:, sl].contiguous(),
                )
                for t in ("query", "key"):
                    nr, _, _ = _relerr_maxmean(gp[t][:, sl], gpi[t])
                    worst[t] = max(worst[t], nr)
                wv_accum += gpi["w_v"]
                sink_accum += gpi["sink"]
            worst["w_v"] = max(
                worst["w_v"], _relerr_maxmean(gp["w_v"], wv_accum)[0]
            )
            worst["sink"] = max(
                worst["sink"], _relerr_maxmean(gp["sink"], sink_accum)[0]
            )
        print(f"\n[A4] mqa_dsa backward norm-rel (saturated): {worst}")
        for t in ("query", "key", "w_v", "sink"):
            self.assertLess(worst[t], 3e-2, f"d{t} packed!=single {worst}")


@_GPU
class TestForcedWindowAndSuperset(unittest.TestCase):
    """Task 2 + Task 3 on the ACTUAL ``token_indices`` handed to the kernel.

    Proves the forced 128-window is always selected and clipped at the document
    boundary, and quantifies the strict full-causal-set relationship for rows
    whose available column count fits the budget.
    """

    def setUp(self):
        _CAPTURED.clear()
        self.module = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.0), bf16=True
        )
        self.module.eval()

    def _capture(self, layout, seqlen):
        query, key, w_v, x, qr = _dsa_inputs(seqlen, seed=4)
        row_end = _row_end(layout, seqlen)
        _CAPTURED.clear()
        self.module(
            query, key, None, None, row_end, v_b_proj_weight=w_v, x=x, qr=qr
        )
        return _CAPTURED[-1], row_end

    def test_window_always_present_and_clipped_sparse(self):
        # s=512, budget=256 -> genuinely sparse for early-doc-position rows.
        for layout in ([200, 312], [512], [500, 12], [128, 128, 128, 128]):
            with self.subTest(layout=layout):
                idx, row_end = self._capture(layout, 512)
                self.assertEqual(idx.shape[-1], WINDOW + INDEX_TOPK)
                _check_index_invariants(self, idx, row_end, 512)

    def test_first_token_of_every_document_sees_only_itself(self):
        idx, row_end = self._capture([40, 88, 128], 256)
        doc_start, is_valid = _doc_meta(row_end, 256)
        starts = sorted({int(s) for s in doc_start.tolist()})
        for st in starts:
            cols = idx[0, st]
            cols = sorted(cols[cols >= 0].tolist())
            self.assertEqual(
                cols, [st], f"doc first token {st} leaks cross-document"
            )

    def test_window_clip_at_boundary_second_doc(self):
        # A query 100 tokens into the 2nd doc must never reach doc 1, even
        # though i-127 would cross the boundary.
        layout, seqlen = [128, 384], 512
        idx, row_end = self._capture(layout, seqlen)
        q = 128 + 100  # 100 into doc 2 (doc_start=128)
        cols = idx[0, q]
        cols = cols[cols >= 0].tolist()
        self.assertTrue(all(c >= 128 for c in cols), "window crossed boundary")
        self.assertIn(128, cols, "doc-2 start must be in the clipped window")

    def test_superset_quantified(self):
        """Task 3: rows with available_count <= budget must select the FULL
        causal set (== mqa exactly); rows above must be strictly sparse."""
        layout, seqlen = [500, 12], 512
        idx, row_end = self._capture(layout, seqlen)
        doc_start, is_valid = _doc_meta(row_end, seqlen)
        budget = WINDOW + INDEX_TOPK
        n_full, n_sparse = 0, 0
        for q in range(seqlen):
            if not is_valid[q]:
                continue
            start = int(doc_start[q])
            avail = q - start + 1
            cols = set(idx[0, q][idx[0, q] >= 0].tolist())
            full = set(range(start, q + 1))
            if avail <= budget:
                self.assertEqual(cols, full, f"row {q}: not full causal set")
                n_full += 1
            else:
                self.assertTrue(cols.issubset(full), f"row {q}: non-causal")
                self.assertEqual(len(cols), budget, f"row {q}: budget unfilled")
                n_sparse += 1
        print(f"\n[A4] superset: full-set rows={n_full} sparse rows={n_sparse}")
        self.assertGreater(n_sparse, 0, "layout did not exercise sparse rows")
        self.assertGreater(n_full, 0, "layout did not exercise full rows")


def _dense_reference_fp64(query, key, w_v, row_end, scale, sink=None):
    """Pure-numpy float64 per-document causal attention with optional sink.

    Independent of paddle ops, so it is a genuine external oracle for the
    softmax normalisation (including the value-less sink column)."""
    seqlen = int(query.shape[1])
    doc_start, is_valid = _doc_meta(row_end, seqlen)
    q = query[0].cast("float32").numpy().astype(np.float64)  # [s,h,dk]
    k = key.squeeze(2)[0].cast("float32").numpy().astype(np.float64)  # [s,dk]
    wv = w_v.cast("float32").numpy().astype(np.float64)  # [dv,h,vd]
    pos = np.arange(seqlen)
    out = np.zeros([seqlen, H, V_HEAD_DIM], dtype=np.float64)
    for i in range(seqlen):
        if not is_valid[i]:
            continue
        cols = [
            j
            for j in range(seqlen)
            if j <= i and j >= doc_start[i] and is_valid[i]
        ]
        for h in range(H):
            logits = np.array(
                [scale * float(q[i, h] @ k[j]) for j in cols], dtype=np.float64
            )
            if sink is not None:
                logits = np.append(logits, float(sink[h]))
            m = logits.max()
            e = np.exp(logits - m)
            probs = e / e.sum()
            real = probs[: len(cols)]
            # normalisation sanity: real + sink mass == 1 exactly in fp64.
            ctx = sum(real[c] * k[cols[c], :DV] for c in range(len(cols)))
            out[i, h] = ctx @ wv[:, h, :]
    return out.reshape([1, seqlen, H * V_HEAD_DIM]), is_valid


@_GPU
class TestSinkNormalizationFP64(unittest.TestCase):
    """Task 4: verify the sink softmax normalisation against an fp64 oracle,
    that the sink is present for the first token of every document, and that it
    is applied exactly once (not duplicated per document)."""

    def test_mqa_sink_matches_fp64_oracle(self):
        seqlen, layout = 256, [40, 88, 128]
        sink = np.linspace(1.0, 3.0, H)
        # bf16 sink: dense FA4 asserts that dtype on ``learnable_sink``
        # (``flash_mask/cute/interface.py:598``), matching production's
        # ``params_dtype``. The oracle is then fed the rounded values the kernel
        # actually saw, so the comparison isolates the softmax arithmetic rather
        # than re-measuring bf16 storage of the sink.
        module = _build_module(_create_mqa_config("mqa"), bf16=True, sink=sink)
        sink = module.softmax_offset.cast("float64").numpy()
        query, key, w_v = _make_inputs(seqlen, seed=6)
        row_end = _row_end(layout, seqlen)
        out = (
            module(query, key, None, None, row_end, v_b_proj_weight=w_v)
            .cast("float32")
            .numpy()
        )
        ref, _ = _dense_reference_fp64(
            query, key, w_v, row_end, module.softmax_scale, sink=sink
        )
        rel, _, _ = _relerr_maxmean(out, ref)
        print(f"\n[A4] mqa sink vs fp64 oracle norm-rel = {rel:.3e}")
        self.assertLess(rel, 5e-3, "kernel sink softmax != fp64 oracle")
        # The sink genuinely drains mass: sinkless oracle is far away.
        ref0, _ = _dense_reference_fp64(
            query, key, w_v, row_end, module.softmax_scale, sink=None
        )
        self.assertGreater(_relerr_maxmean(ref0, ref)[0], 5e-2)

    def test_sink_not_duplicated_per_document(self):
        """One shared [H] logit -> the drained mass on a first token depends
        only on the (single) sink value, identical for every document's first
        token given identical single-element causal sets. A per-document
        duplication would scale the sink mass by the document index."""
        seqlen, layout = 256, [50, 50, 50, 106]
        sink = np.full(H, 20.0)  # dominant -> first-token output ~ 0
        module = _build_module(_create_mqa_config("mqa"), bf16=True, sink=sink)
        query, key, w_v = _make_inputs(seqlen, seed=6)
        row_end = _row_end(layout, seqlen)
        out = (
            module(query, key, None, None, row_end, v_b_proj_weight=w_v)
            .cast("float32")
            .numpy()
        )
        doc_start, _ = _doc_meta(row_end, seqlen)
        for st in sorted({int(s) for s in doc_start.tolist()}):
            self.assertLess(
                float(np.abs(out[0, st]).max()),
                1e-2,
                f"doc first token {st}: sink missing or mis-scaled",
            )


@_GPU
class TestMTPDocumentMaskShift(unittest.TestCase):
    """Task 6: the MTP layer shares the exact masking code of the main layers
    (``is_mtp_layer`` is a pure no-op here). The MTP-specific token shift is
    applied entirely upstream (``multi_token_prediction`` slices
    ``mtp_startend_row_indices_all[:, depth]`` and passes it as the ordinary
    ``attn_mask_startend_row_indices``); the per-depth mask is still a
    well-formed per-document end-row mask. So MTP isolation reduces to correct
    handling of a (shifted) document mask, which we verify directly here.

    A depth-``k`` shift shrinks every document by ``k`` tokens; we model that
    with the shrunk, re-expressed layout and require packed==single and
    in-document-only selection on both the dense and the DSA path.
    """

    _BASE = [40, 88, 128]
    _SEQ = 256

    def _shifted(self, depth):
        return [max(L - depth, 1) for L in self._BASE]

    def test_is_mtp_layer_is_a_noop(self):
        query, key, w_v = _make_inputs(self._SEQ, seed=6)
        row_end = _row_end(self._BASE, self._SEQ)
        main = _build_module(_create_mqa_config("mqa"), is_mtp=False)
        mtp = _build_module(_create_mqa_config("mqa"), is_mtp=True)
        o_main = main(query, key, None, None, row_end, v_b_proj_weight=w_v)
        o_mtp = mtp(query, key, None, None, row_end, v_b_proj_weight=w_v)
        self.assertEqual(
            float((o_main.cast("float32") - o_mtp.cast("float32")).abs().max()),
            0.0,
            "is_mtp_layer changed the masking/output",
        )

    def test_shifted_mask_packed_equals_single_mqa(self):
        module = _build_module(_create_mqa_config("mqa"), is_mtp=True)
        query, key, w_v = _make_inputs(self._SEQ, seed=6)
        for depth in (1, 2):
            with self.subTest(depth=depth):
                layout = self._shifted(depth)
                row_end = _row_end(layout, self._SEQ)
                packed = (
                    module(query, key, None, None, row_end, v_b_proj_weight=w_v)
                    .cast("float32")
                    .numpy()
                )
                ref = _dense_reference(
                    query, key, w_v, row_end, module.softmax_scale
                ).numpy()
                self.assertLess(_relerr_maxmean(packed, ref)[0], 5e-3)
                for start, length in _doc_segments(layout, self._SEQ):
                    sl = slice(start, start + length)
                    piece = (
                        module(
                            query[:, sl].contiguous(),
                            key[:, sl].contiguous(),
                            None,
                            None,
                            _row_end([length], length),
                            v_b_proj_weight=w_v,
                        )
                        .cast("float32")
                        .numpy()
                    )
                    _assert_agrees_to_bf16_ulps(
                        self,
                        packed[:, sl],
                        piece,
                        f"depth {depth} shifted doc@{start} not isolated",
                    )

    def test_shifted_mask_no_cross_doc_in_dsa(self):
        module = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.0), bf16=True
        )
        module.eval()
        query, key, w_v, x, qr = _dsa_inputs(self._SEQ, seed=6)
        row_end = _row_end(self._shifted(1), self._SEQ)
        _CAPTURED.clear()
        module(query, key, None, None, row_end, v_b_proj_weight=w_v, x=x, qr=qr)
        _check_index_invariants(self, _CAPTURED[-1], row_end, self._SEQ)


if __name__ == "__main__":
    unittest.main()
