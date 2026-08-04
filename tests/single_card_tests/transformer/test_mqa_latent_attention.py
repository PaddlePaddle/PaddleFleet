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

"""Unit tests for :mod:`paddlefleet.transformer.mqa_latent_attention`.

``non_absorbed_mqa=True`` turns the hybrid MLA (``csa_compress_ratios == -2``)
layers of a ``dsv4_hybrid`` model into :class:`MQALatentAttention`. The module
picks its path from the sublayers spec, not from any config string:

* ``MQALatentAttentionSublayersSpec(indexer=None)`` -- per-document full-causal
  dense attention on the latent, mathematically equal to MHA. In production the
  indexer is always built (``gpt_layer_specs`` when ``non_absorbed_mqa`` is set),
  so this indexer-less path exists only for the absorption-equivalence tests
  here, driven by constructing the layer directly with ``indexer=None``.
* an indexer spec -- forced local window + Lightning-indexer top-k, i.e. DSA on
  the KV latent.

Coverage:
  1. Guards -- unsupported configurations fail loudly (no GPU needed).
  2. Index construction over adversarial multi-document layouts: the forced
     128-window and the indexer candidate range are disjoint yet jointly equal
     the per-document causal set (no duplicate column, no lost window column).
  3. The indexer-less dense path equals a dense fp32 reference, because the
     activation-level absorption is exactly score preserving.
  4. Packed multi-document equals independent per-document runs, bit exact.
  5. The DSA (indexer) path: a saturated budget reproduces the dense
     reference, a sparse budget stays causal/duplicate-free, backward yields
     finite gradients, and the recompute double-forward selects identical
     columns while attaching the indexer loss on the grad-enabled pass only.
  6. The model-wide learnable per-head sink (``add_full_attention_sink_bias`` /
     ``softmax_type``, built by the shared ``build_softmax_offset`` helper)
     equals one extra value-less softmax column (on both the dense and the DSA
     path) and takes a finite non-zero fp32 gradient. There is a single sink
     switch now, so the config no longer rejects any combination.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.csa_attention import (
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
)
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.transformer_config import TransformerConfig

from .hybrid_mla_utils import (
    _CAPTURED,
    _GPU,
    DK,
    DV,
    HIDDEN,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    INDEX_TOPK,
    K_CHANNELS,
    Q_LORA,
    V_HEAD_DIM,
    WINDOW,
    H,
    _build_module,
    _check_index_invariants,
    _create_mqa_config,
    _dense_reference,
    _make_inputs,
    _rel,
    _row_end,
)

# Adversarial document layouts: shorter than / equal to / longer than the
# forced window, single-token documents, and a document overrunning the buffer.
_LAYOUTS = [
    [1, 2, 3],
    [5, 7],
    [WINDOW],
    [WINDOW + 1],
    [WINDOW - 1, 2],
    [WINDOW + 2, 1],
    [8],
    [1, 1, 1, 1, 1, 1],
    [3, WINDOW, WINDOW + 1, 1],
    [300],
]


class TestMQAGuards(unittest.TestCase):
    """Unsupported configurations must fail loudly, not silently mis-compute."""

    def setUp(self):
        self.module = _build_module(_create_mqa_config("mqa"))

    @staticmethod
    def _args(b=1, s=8):
        query = paddle.zeros([b, s, H, DK], dtype="bfloat16")
        key = paddle.zeros([b, s, 1, DK], dtype="bfloat16")
        w_v = paddle.zeros([DV, H, V_HEAD_DIM], dtype="bfloat16")
        return query, key, w_v

    def test_packed_seq_params_rejected(self):
        query, key, w_v = self._args()
        with self.assertRaises(NotImplementedError):
            self.module(
                query,
                key,
                None,
                None,
                packed_seq_params=object(),
                v_b_proj_weight=w_v,
            )

    def test_missing_v_b_proj_weight_rejected(self):
        query, key, _ = self._args()
        with self.assertRaises(ValueError):
            self.module(query, key, None, None)

    def test_batch_size_gt_one_rejected(self):
        query, key, w_v = self._args(b=2)
        with self.assertRaises(NotImplementedError):
            self.module(query, key, None, None, v_b_proj_weight=w_v)

    def test_softmax_scale_is_the_mha_scale(self):
        # Absorption is exactly score preserving, so the scale must stay the MHA
        # q_head_dim one (256**-0.5), never the 576-wide latent one.
        self.assertAlmostEqual(
            self.module.softmax_scale, K_CHANNELS**-0.5, places=12
        )
        self.assertAlmostEqual(self.module.softmax_scale, 0.0625, places=12)


class TestMQAIndexRanges(unittest.TestCase):
    """The forced window and the indexer candidate range partition the causal
    set: no overlap (would double-count) and no gap (would waste budget)."""

    def setUp(self):
        self.module = _build_module(_create_mqa_config("mqa"))

    def test_window_and_indexer_range_partition_causal_set(self):
        seqlen = 256
        for layout in _LAYOUTS:
            with self.subTest(layout=layout):
                row_end = _row_end(layout, seqlen)
                doc_start, doc_len, is_valid, _, _ = _derive_csa_doc_boundaries(
                    row_end, seqlen
                )
                window = _build_window_topk_idxs_from_doc_bounds(
                    1, seqlen, WINDOW, doc_start, is_valid
                ).numpy()
                valid_range, row_empty = self.module._indexer_valid_range(
                    seqlen, doc_start, doc_len, is_valid
                )
                self._assert_partition(
                    window,
                    valid_range.numpy()[0],
                    row_empty.numpy().reshape([seqlen]),
                    doc_start.numpy(),
                    is_valid.numpy(),
                    seqlen,
                    layout,
                )

    def _assert_partition(
        self, window, vr, row_empty, doc_start, is_valid, seqlen, layout
    ):
        for q in range(seqlen):
            win = {int(c) for c in window[0, q] if c >= 0}
            cand = set(range(int(vr[q, 0]), int(vr[q, 1])))
            tag = f"{layout} row {q}"
            if not is_valid[q]:
                self.assertEqual(win, set(), tag)
                self.assertEqual(cand, set(), tag)
                self.assertTrue(bool(row_empty[q]), tag)
                continue
            start = int(doc_start[q])
            self.assertEqual(
                win, set(range(max(start, q - WINDOW + 1), q + 1)), tag
            )
            self.assertEqual(win & cand, set(), f"{tag}: overlap")
            self.assertEqual(
                win | cand, set(range(start, q + 1)), f"{tag}: incomplete"
            )
            self.assertEqual(bool(row_empty[q]), not cand, tag)


class TestHybridMLAConfig(unittest.TestCase):
    """The hybrid MLA config surface after the ``non_absorbed_mqa`` refactor.

    The old 3-state ``hybrid_mla_attn_mode`` and the ``hybrid_mla_attn_sink``
    switch (with its mutual-exclusion ValueError against the model-wide sinks)
    are gone. There is now a single boolean ``non_absorbed_mqa`` and a single
    sink switch (``add_full_attention_sink_bias`` / ``softmax_type``), so the
    two can no longer conflict. When ``non_absorbed_mqa=True`` the -2 layers run
    a cuDNN DSA indexer, so the config validates the model-wide ``dsa_index_*``
    fields (index_n_heads / index_head_dim / index_topk).
    """

    @staticmethod
    def _kwargs(**overrides):
        kwargs = {
            "num_hidden_layers": 2,
            "hidden_size": HIDDEN,
            "num_attention_heads": H,
            "experimental_attention_variant": "dsv4_hybrid",
            "csa_compress_ratios": [-2, -2],
            "hybrid_mla_q_lora_rank": Q_LORA,
            "hybrid_mla_kv_lora_rank": DV,
            "hybrid_mla_qk_nope_head_dim": 192,
            "hybrid_mla_qk_rope_head_dim": 64,
            "hybrid_mla_v_head_dim": V_HEAD_DIM,
            "hybrid_mla_num_attention_heads": H,
            "hybrid_mla_num_key_value_heads": H,
        }
        kwargs.update(overrides)
        return kwargs

    @classmethod
    def _non_absorbed_kwargs(cls, **overrides):
        # ``non_absorbed_mqa=True`` triggers the DSA-indexer validation, so a
        # valid baseline must carry the model-wide index dims.
        base = {
            "non_absorbed_mqa": True,
            "dsa_index_n_heads": INDEX_HEADS,
            "dsa_index_head_dim": INDEX_HEAD_DIM,
            "dsa_index_topk": INDEX_TOPK,
        }
        base.update(overrides)
        return cls._kwargs(**base)

    def test_non_absorbed_mqa_defaults_off(self):
        config = TransformerConfig(**self._kwargs())
        self.assertFalse(config.non_absorbed_mqa)

    def test_non_absorbed_mqa_true_accepted_with_valid_index_dims(self):
        config = TransformerConfig(**self._non_absorbed_kwargs())
        self.assertTrue(config.non_absorbed_mqa)
        self.assertEqual(config.dsa_index_head_dim, 128)

    def test_sink_coexists_with_non_absorbed_mqa(self):
        # The old mutual-exclusion ValueError is gone: one sink switch only, so
        # enabling a model-wide sink alongside non_absorbed_mqa must be accepted.
        for sink in (
            {"add_full_attention_sink_bias": True},
            {"softmax_type": "learnable"},
        ):
            with self.subTest(sink=sink):
                config = TransformerConfig(**self._non_absorbed_kwargs(**sink))
                self.assertTrue(config.non_absorbed_mqa)

    def test_index_dims_are_validated(self):
        # Keyed on (field, value), not on the message: topk=100 (not a multiple
        # of 128) and topk=2176 (>2048) both mention "index_topk" but hit
        # different branches.
        for field, value, msg in (
            ("dsa_index_head_dim", 64, "index_head_dim"),
            ("dsa_index_topk", 100, "index_topk"),
            ("dsa_index_topk", 2048 + 128, "index_topk"),
            ("dsa_index_n_heads", None, "index_n_heads"),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaisesRegex(ValueError, msg),
            ):
                TransformerConfig(**self._non_absorbed_kwargs(**{field: value}))

    def test_dense_switch_alone_is_rejected(self):
        # ``non_absorbed_mqa_dense`` replaces that mode's indexer, so on its own
        # it would be a silent no-op. Must be a config error instead.
        with self.assertRaisesRegex(ValueError, "non_absorbed_mqa_dense"):
            TransformerConfig(**self._kwargs(non_absorbed_mqa_dense=True))

    def test_dense_switch_does_not_require_index_dims(self):
        # No indexer is built, so the index_* validation must be skipped -- these
        # kwargs deliberately omit dsa_index_* and would otherwise be rejected.
        config = TransformerConfig(
            **self._kwargs(non_absorbed_mqa=True, non_absorbed_mqa_dense=True)
        )
        self.assertTrue(config.non_absorbed_mqa_dense)

    def test_index_dims_unvalidated_when_non_absorbed_mqa_off(self):
        """With the flag off the -2 layers are dense MHA, so the indexer fields
        are unused and must not be validated at all.

        Asserting only that a *default* config builds is near-tautological: the
        defaults are ``None``, which is exactly what
        ``test_index_dims_are_validated`` shows the flag-on path rejects, but
        nothing pins the other three rejections. So feed the exact values that
        test proves are rejected when the flag is on -- head_dim != 128, topk
        not a multiple of 128, topk > 2048 -- and assert each one builds and
        survives onto the config unchanged.
        """
        bad = {
            "dsa_index_n_heads": None,
            "dsa_index_head_dim": 64,
            "dsa_index_topk": 100,
        }
        for field, value in [*bad.items(), ("dsa_index_topk", 2048 + 128)]:
            with self.subTest(field=field, value=value):
                config = TransformerConfig(
                    **self._kwargs(non_absorbed_mqa=False, **{field: value})
                )
                self.assertFalse(config.non_absorbed_mqa)
                self.assertEqual(getattr(config, field), value)
        # ... and all of them together, still no raise.
        config = TransformerConfig(
            **self._kwargs(non_absorbed_mqa=False, **bad)
        )
        self.assertFalse(config.non_absorbed_mqa)


@_GPU
class TestMQAEquivalence(unittest.TestCase):
    """The indexer-less dense path is mathematically identical to MHA."""

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def setUp(self):
        _CAPTURED.clear()
        self.module = _build_module(_create_mqa_config("mqa"))

    def _run(self, seqlen, layout):
        query, key, w_v = _make_inputs(seqlen)
        row_end = None if layout is None else _row_end(layout, seqlen)
        out = self.module(query, key, None, None, row_end, v_b_proj_weight=w_v)
        ref = _dense_reference(
            query,
            key,
            w_v,
            _row_end([seqlen], seqlen) if row_end is None else row_end,
            self.module.softmax_scale,
        )
        return out, ref

    def test_single_document_matches_dense(self):
        out, ref = self._run(256, None)
        # W7 measured _rel 2.20e-3..2.63e-3 over seeds 0-4 (bf16, cudnn
        # deterministic); tightened 5e-3 -> 3.5e-3 (1.3x headroom over worst).
        self.assertLess(_rel(out, ref), 3.5e-3)

    def test_multi_document_layouts_match_dense(self):
        seqlen = 256
        for layout in _LAYOUTS:
            with self.subTest(layout=layout):
                _CAPTURED.clear()
                out, ref = self._run(seqlen, layout)
                # W7 measured max _rel 2.63e-3 over 10 layouts x seeds 0-4;
                # tightened 5e-3 -> 3.5e-3.
                self.assertLess(_rel(out, ref), 3.5e-3)
                _check_index_invariants(
                    self,
                    _CAPTURED[-1],
                    _row_end(layout, seqlen),
                    seqlen,
                    expect_full=True,
                )

    def test_attention_sink_matches_dense_reference(self):
        """The learnable sink is one extra value-less softmax column."""
        seqlen, layout = 256, [40, 88, 128]
        sink = np.linspace(1.0, 3.0, H)
        module = _build_module(_create_mqa_config("mqa"), sink=sink)
        self.assertEqual(module.softmax_offset.dtype, paddle.float32)
        self.assertEqual(list(module.softmax_offset.shape), [H])
        query, key, w_v = _make_inputs(seqlen)
        row_end = _row_end(layout, seqlen)
        out = module(query, key, None, None, row_end, v_b_proj_weight=w_v)
        sink_used = module.softmax_offset.astype("float32").numpy()
        ref = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale, sink=sink_used
        )
        # W7 measured _rel 2.74e-3 over seeds 0-2 (dense + sink); 5e-3 -> 3.5e-3.
        self.assertLess(_rel(out, ref), 3.5e-3)
        # ... and the sink genuinely changed the result: a positive logit drains
        # probability mass, so the sinkless reference is far away.
        sinkless = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale
        )
        # W7 measured sinkless _rel ~0.985-1.02 (the sink dominates this layout);
        # lower-bound tightened 5e-2 -> 0.5 (2x headroom below worst observed).
        self.assertGreater(_rel(sinkless, ref), 0.5)


@_GPU
class TestMQADSA(unittest.TestCase):
    """The DSA (indexer) path: forced window + Lightning-indexer top-k."""

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        self.module = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.01), bf16=True
        )

    @staticmethod
    def _inputs(seqlen, seed=0):
        return _make_inputs(seqlen, seed=seed, with_hidden=True)

    def _forward(self, seqlen, layout, seed=0):
        """Inference-mode forward, for checking numerics and index invariants.

        ``eval()`` skips the indexer KL loss, which is irrelevant here and
        cannot be attached to an all-detached graph: the autoscaler PyLayer
        returns its ``output`` argument unchanged, and Paddle rejects that
        "inplace" return for a leaf tensor. Real training always feeds a
        differentiable ``query``; the loss path is covered by the backward and
        recompute tests below.
        """
        query, key, w_v, x, qr = self._inputs(seqlen, seed=seed)
        self.module.eval()
        row_end = _row_end(layout, seqlen)
        out = self.module(
            query,
            key,
            None,
            None,
            row_end,
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )
        return out, (query, key, w_v, x, qr), row_end

    def test_saturated_budget_reproduces_dense(self):
        # window(128) + topk(128) covers every row's causal length at s=256, so
        # the selected set is exactly the full causal set and DSA must be exact.
        seqlen = WINDOW + INDEX_TOPK
        for layout in ([seqlen], [40, 88, 128], [3, WINDOW, WINDOW + 1, 1]):
            with self.subTest(layout=layout):
                _CAPTURED.clear()
                out, tensors, row_end = self._forward(seqlen, layout)
                query, key, w_v = tensors[0], tensors[1], tensors[2]
                ref = _dense_reference(
                    query, key, w_v, row_end, self.module.softmax_scale
                )
                # W7 measured max _rel 2.62e-3 over 3 layouts x seeds 0-2
                # (DSA saturated budget); 5e-3 -> 3.5e-3.
                self.assertLess(_rel(out, ref), 3.5e-3)
                self.assertEqual(_CAPTURED[-1].shape[-1], WINDOW + INDEX_TOPK)
                _check_index_invariants(
                    self, _CAPTURED[-1], row_end, seqlen, expect_full=True
                )

    def test_sparse_budget_indices_are_sound(self):
        seqlen = 512  # window + topk = 256 < 512 => genuinely sparse
        _CAPTURED.clear()
        out, _, row_end = self._forward(seqlen, [200, 312])
        self.assertTrue(bool(paddle.isfinite(out.cast("float32")).all()))
        self.assertEqual(_CAPTURED[-1].shape[-1], WINDOW + INDEX_TOPK)
        _check_index_invariants(self, _CAPTURED[-1], row_end, seqlen)

    def test_backward_produces_finite_grads_and_reports_loss(self):
        seqlen = WINDOW + INDEX_TOPK
        query, key, w_v, x, qr = self._inputs(seqlen)
        for tensor in (query, key, x, qr):
            tensor.stop_gradient = False
        self.module.train()
        out = self.module(
            query,
            key,
            None,
            None,
            _row_end([seqlen], seqlen),
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )
        out.cast("float32").sum().backward()
        for name, tensor in (("query", query), ("key", key)):
            self.assertIsNotNone(tensor.grad, f"{name} has no gradient")
            self.assertTrue(
                bool(paddle.isfinite(tensor.grad.cast("float32")).all()),
                f"{name} gradient is not finite",
            )
        # The indexer inputs are deliberately detached from the backbone (same
        # contract as DSAttention/CSA): the indexer learns from its own KL loss
        # only, so ``x``/``qr`` must stay gradient-free while the indexer
        # projections still receive gradients.
        self.assertIsNone(x.grad)
        self.assertIsNone(qr.grad)
        indexer_params = {
            "wq_b": self.module.indexer.wq_b.linear.weight,
            "wk": self.module.indexer.wk.linear.weight,
            "weights_proj": self.module.indexer.weights_proj.linear.weight,
        }
        for name, param in indexer_params.items():
            self.assertIsNotNone(param.grad, f"indexer.{name} has no gradient")
            self.assertTrue(
                bool(paddle.isfinite(param.grad.cast("float32")).all()),
                f"indexer.{name} gradient is not finite",
            )
            self.assertGreater(
                float(param.grad.cast("float32").abs().max()),
                0.0,
                f"indexer.{name} gradient is all zero",
            )
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)

    def test_indexer_loss_mask_comes_from_input_ids(self):
        """Padding rows must not dilute the KL loss, and only ``input_ids`` sees them.

        Same masked reduction and the same mask construction as
        ``csa_attention.py:1302-1306`` / ``:2411-2443``. The document metadata
        cannot stand in for it: ``attn_mask_startend_row_indices`` has no way to
        say "these trailing rows are padding", so the tail here is folded into a
        second document and every row reports ``is_valid == True``. Only
        ``input_ids != pad_token_id`` separates them.

        Two runs share the same 384-token document and differ only in how much
        trailing padding follows it. With the mask both must report the same KL;
        without it (a plain ``mean()``, or a mask derived from ``is_valid``) the
        pad rows enter both numerator and denominator and the two diverge.
        """
        doc = 384  # > WINDOW, so later rows have a non-empty top-k set
        query, key, w_v, x, qr = self._inputs(512, seed=3)

        def run(seqlen):
            DSAIndexerLossLoggingHelper.tracker.clear()
            tensors = [t[:, :seqlen].clone() for t in (query, key, x, qr)]
            for tensor in tensors:
                tensor.stop_gradient = False
            # 1 = real token, 0 = pad (``config.pad_token_id`` defaults to 0).
            input_ids = paddle.concat(
                [
                    paddle.ones([1, doc], dtype="int64"),
                    paddle.zeros([1, seqlen - doc], dtype="int64"),
                ],
                axis=-1,
            )
            self.module.train()
            out = self.module(
                tensors[0],
                tensors[1],
                None,
                None,
                _row_end([doc, seqlen - doc], seqlen),
                v_b_proj_weight=w_v,
                x=tensors[2],
                qr=tensors[3],
                input_ids=input_ids,
            )
            out.cast("float32").sum().backward()
            return float(DSAIndexerLossLoggingHelper.tracker["values"][0])

        loss_448, loss_512 = run(448), run(512)
        self.assertGreater(loss_448, 0.0)
        # W7 measured loss_512/loss_448 - 1 == 0.0 exactly over seeds
        # 0/3/4/7/11 (two-doc layout is bit-reproducible); delta 2e-3 -> 5e-4.
        self.assertAlmostEqual(loss_512 / loss_448, 1.0, delta=5e-4)

    def test_use_sparse_loss_widens_only_the_loss_topk(self):
        """``dsa_indexer_use_sparse_loss`` picks the KL width, not attention's.

        Same contract as the CSA layers of the same model
        (``_resolve_csa_indexer_loss_topk_effective``): ``False`` scores a wider
        candidate table so a fresh indexer is also supervised on columns it did
        not pick, while attention keeps consuming ``window + index_topk``
        unchanged. ``s=512`` widens the loss table 128 -> 512.

        On this single full-length document neither the output bits nor the
        index table are reproducible across identical calls: measured 2.4-2.8%
        of the table's slots move between two *identical* eval-mode calls, and
        ~1e-4 on the output. (Splitting the same 512 rows into two documents is
        exactly reproducible, which is why
        ``test_recompute_double_forward_is_consistent`` can assert equality --
        it uses ``[200, 312]``.) So the loss width is asserted exactly -- it is
        an integer the switch controls directly -- and the attention side is
        asserted against that same-setting drift as a baseline. The failure
        mode this rules out is prefix-truncating the wide table into attention,
        which would move ~37% of the slots (the kernel emits ascending
        *position* order, not score order).
        """
        seqlen = 512
        query, key, w_v, x, qr = self._inputs(seqlen, seed=5)
        loss_widths = []
        inner_target = self.module._attn_target

        def recording_target(query_, kv_, topk_indices):
            loss_widths.append(int(topk_indices.shape[-1]))
            return inner_target(query_, kv_, topk_indices)

        self.module._attn_target = recording_target

        def run(sparse):
            _CAPTURED.clear()
            DSAIndexerLossLoggingHelper.tracker.clear()
            tensors = [t.clone() for t in (query, key, x, qr)]
            for tensor in tensors:
                tensor.stop_gradient = False
            self.module.indexer_use_sparse_loss = sparse
            self.module.train()
            out = self.module(
                tensors[0],
                tensors[1],
                None,
                None,
                _row_end([seqlen], seqlen),
                v_b_proj_weight=w_v,
                x=tensors[2],
                qr=tensors[3],
            )
            out.cast("float32").sum().backward()
            return (
                _CAPTURED[-1].copy(),
                float(DSAIndexerLossLoggingHelper.tracker["values"][0]),
            )

        idx_a, loss_sparse = run(True)
        idx_b, _ = run(True)
        idx_full, loss_full = run(False)

        # Only the KL table widened; attention kept its own budget.
        self.assertEqual(loss_widths, [INDEX_TOPK, INDEX_TOPK, seqlen])
        for table in (idx_a, idx_b, idx_full):
            self.assertEqual(int(table.shape[-1]), WINDOW + INDEX_TOPK)
        baseline = float((idx_a != idx_b).mean())
        cross = float((idx_full != idx_b).mean())
        # W7 measured cross/baseline ~0.90-1.02 over 3 trials (cross tracks the
        # identical-call drift), while the failure it guards -- prefix-truncating
        # the wide loss table into attention -- moves ~37% of slots. 4x -> 3x
        # still cleanly separates ~2.7% (pass) from 37% (fail).
        self.assertLess(cross, max(3 * baseline, 0.02))
        self.assertGreater(loss_sparse, 0.0)
        self.assertGreater(loss_full, 0.0)
        # A wider renormalisation set means a different KL; equal values would
        # mean the switch never reached the loss.
        self.assertGreater(abs(loss_full - loss_sparse), 1e-6)

    def test_recompute_double_forward_is_consistent(self):
        """Reentrant recompute runs the layer twice: pass 1 under ``no_grad``.

        The top-k must be deterministic across the two passes (otherwise the
        backward would differentiate a different sparsity pattern), and the
        indexer loss must be attached on the grad-enabled pass only.
        """
        seqlen = 512
        query, key, w_v, x, qr = self._inputs(seqlen)
        query.stop_gradient = False
        self.module.train()
        row_end = _row_end([200, 312], seqlen)
        kwargs = {"v_b_proj_weight": w_v, "x": x, "qr": qr}

        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        with paddle.no_grad():
            self.module(query, key, None, None, row_end, **kwargs)
        first = _CAPTURED[-1]
        self.assertNotIn(
            "values",
            DSAIndexerLossLoggingHelper.tracker,
            "indexer loss must not be attached on the no_grad pass",
        )

        self.module(query, key, None, None, row_end, **kwargs)
        second = _CAPTURED[-1]
        np.testing.assert_array_equal(first, second)
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)

    def test_sink_saturated_budget_matches_dense(self):
        """DSA + sink: the finite-sink LSE correction must be exact.

        ``d_qk``(576) != ``d_v``(512) here, so the kernel takes its finite-sink
        correction path; a saturated budget makes the selected set the full
        causal set, hence the dense sink reference applies exactly.
        """
        seqlen = WINDOW + INDEX_TOPK
        sink = np.linspace(1.0, 3.0, H)
        module = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.01),
            bf16=True,
            sink=sink,
        )
        module.eval()
        query, key, w_v, x, qr = self._inputs(seqlen)
        row_end = _row_end([40, 88, 128], seqlen)
        out = module(
            query,
            key,
            None,
            None,
            row_end,
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )
        sink_used = module.softmax_offset.astype("float32").numpy()
        ref = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale, sink=sink_used
        )
        # W7 measured _rel 2.75e-3 over seeds 0-1 (DSA + finite-sink LSE
        # correction, saturated budget); 5e-3 -> 3.5e-3.
        self.assertLess(_rel(out, ref), 3.5e-3)
        sinkless = _dense_reference(
            query, key, w_v, row_end, module.softmax_scale
        )
        # W7 measured sinkless _rel ~0.985; lower-bound 5e-2 -> 0.5.
        self.assertGreater(_rel(sinkless, ref), 0.5)

    def test_sink_receives_finite_nonzero_fp32_gradient(self):
        """The sink gradient is computed analytically (the kernel returns 0)."""
        seqlen = WINDOW + INDEX_TOPK
        module = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.01),
            bf16=True,
            sink=np.linspace(1.0, 3.0, H),
        )
        module.train()
        query, key, w_v, x, qr = self._inputs(seqlen)
        for tensor in (query, key, x, qr):
            tensor.stop_gradient = False
        out = module(
            query,
            key,
            None,
            None,
            _row_end([seqlen], seqlen),
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )
        out.cast("float32").sum().backward()
        grad = module.softmax_offset.grad
        self.assertIsNotNone(grad, "attention sink has no gradient")
        # The gradient must come back in the parameter's dtype (bf16 here, the
        # production ``params_dtype``); an fp32 grad on a bf16 parameter would
        # be a dtype mismatch on accumulation.
        self.assertEqual(grad.dtype, module.softmax_offset.dtype)
        self.assertEqual(list(grad.shape), [H])
        self.assertTrue(
            bool(paddle.isfinite(grad.astype("float32")).all()),
            "sink gradient is not finite",
        )
        self.assertGreater(
            float(grad.astype("float32").abs().max()),
            0.0,
            "sink gradient is all zero",
        )
        # The backbone must keep flowing with the sink enabled.
        self.assertIsNotNone(query.grad)
        self.assertTrue(bool(paddle.isfinite(query.grad.cast("float32")).all()))


if __name__ == "__main__":
    unittest.main()
