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

"""Recompute / MTP / RoPE regression for the phase-2 (warmup) shape of
``hybrid_mla_attention="mqa_dsa"``.

The warmup shape is selected by ``dsa_indexer_use_sparse_loss=False``: attention
consumes the full per-document causal table while the indexer's top-k feeds the
wide KL loss only. ``test_mqa_latent_attention.TestMQADSAWarmupPhase`` covers the
single-forward behaviour of that path; this module covers the three axes that
compose *around* it and were previously only exercised in phase 3
(``dsa_indexer_use_sparse_loss=True``):

1. **Recompute** (``TestWarmupRecompute``) -- the real production wrapping,
   ``paddle.distributed.fleet.utils.recompute`` around
   ``MQALatentAttention.forward``, which is how ``full_recompute`` wraps
   ``_forward_impl``. ON must equal OFF, the ``token_indices`` must be
   re-derived bit-identically, and the indexer loss must be attached exactly
   once, on the grad-enabled pass. Warmup is strictly stronger than phase 3
   here: the table is ``_build_full_causal_indices``, a pure integer function of
   the document bounds, so it is bit-identical even on the single-document
   layout whose top-k order phase 3 cannot reproduce. It also *skips the indexer
   entirely* on the ``no_grad`` pass (the ``mqa_latent_attention.py:495`` early
   exit), which phase 3 does not.
2. **MTP** (``TestWarmupMTP``) -- ``MultiTokenPredictionLayer`` builds its
   ``transformer_layer`` without passing ``pg_collection``, so the MTP ``-2``
   layer is the same ``MQALatentAttention`` class with the same config; the
   warmup shape must therefore be live there too. Plus the tracker denominator:
   ``track_indexer_metrics`` now takes the enum string and must count the MTP
   ``-2`` entry of ``csa_compress_ratios``.
3. **RoPE** (``TestWarmupRope``) -- the switch must not touch RoPE. The main
   attention's rotary application happens in ``MLASelfAttention`` before the
   ``mqa_latent`` branch, and the DSA indexer keeps its own plain-RoPE
   (``dsa_indexer_rotary_interleaved``) which is still evaluated in warmup even
   though attention does not consume its ranking. Also a negative case for the
   construction-time ``apply_rope_fusion`` x latent-MQA exclusion.

Every RoPE assertion is against the independent fp64 reference of
``test_hybrid_mla_rope_audit``, never against the implementation itself.

``TestRecomputeInnerForwardBitIdentical`` closes the one gap axis 1 leaves open:
recompute-ON vs recompute-OFF says nothing about whether the *two* forwards of a
recomputed step agree with each other, since paddle discards the first one's
output. That class keeps both and requires ``maxabs == 0.0``, on all three
``hybrid_mla_attention`` shapes and on a ``seqlen`` where the sparse budget does
not already cover the whole causal range.
"""

import unittest

import numpy as np
import paddle

from paddlefleet.transformer.dot_product_attention import DotProductAttention
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSublayersSpec,
)

from .hybrid_mla_utils import (
    _CAPTURED,
    _GPU,
    HIDDEN,
    INDEX_TOPK,
    Q_LORA,
    BiasedLinear,
    LayerNormStub,
    _build_module,
    _check_index_invariants,
    _create_mqa_config,
    _make_inputs,
    _rel,
    _row_end,
)
from .test_hybrid_mla_rope_audit import (
    ROPE_THETA,
    ref_angles,
    ref_inv_freq,
    ref_rope_halfsplit,
)
from .test_mqa_latent_attention import _fp32, _full_causal_table

SEQLEN = 256
# Two documents, the second longer than the forced window (so the indexer's
# candidate range is non-empty), plus the single full-length document that the
# phase-3 recompute test cannot use because its top-k emission order drifts.
TWO_DOCS = [40, 216]
ONE_DOC = [SEQLEN]

# Production 44-slot ratios: -2 at [8, 17, 26, 34, 42, 43]; 43 is the MTP layer.
_PROD_MINUS2 = [8, 17, 26, 34, 42, 43]


def _prod_csa_ratios():
    ratios = [128] * 8 + [-2] + [128] * 8 + [-2] + [128] * 8 + [-2]
    ratios += [128] * 7 + [-2] + [128] * 7 + [-2, -2]
    assert len(ratios) == 44
    assert [i for i, v in enumerate(ratios) if v == -2] == _PROD_MINUS2
    return ratios


def _warmup_config(loss_coeff=0.01, **overrides):
    """A ``"mqa_dsa"`` config with the phase-2 switch off from construction."""
    config = _create_mqa_config("mqa_dsa", loss_coeff=loss_coeff, **overrides)
    config.dsa_indexer_use_sparse_loss = False
    return config


def _leaf(tensor):
    out = tensor.clone().detach()
    out.stop_gradient = False
    return out


def _grad_rel(g_on, g_off):
    if g_on is None and g_off is None:
        return 0.0
    assert (g_on is None) == (g_off is None), "one grad present, the other None"
    return _rel(g_on, g_off)


def _tracker_value(slot):
    values = DSAIndexerLossLoggingHelper.tracker.get("values")
    return None if values is None else float(values.numpy()[slot])


@_GPU
class TestWarmupRecompute(unittest.TestCase):
    """``recompute(MQALatentAttention.forward)`` in the warmup phase.

    This is the production wrapping: ``full_recompute`` hands
    ``TransformerLayer._forward_impl`` to ``paddle.distributed.fleet.utils.
    recompute``, whose reentrant implementation runs the wrapped callable once
    under ``no_grad`` (to produce the output) and once more with grad enabled
    during backward. Both forwards must agree on the sparsity pattern, or the
    backward differentiates a set of columns the forward never used.
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
        self.module = _build_module(_warmup_config(), bf16=True)
        self.assertIsNotNone(self.module.indexer)
        self.assertFalse(self.module.indexer_use_sparse_loss)

    def _run(self, module, row_end, use_recompute, seed=7):
        """One train step, with or without recompute. Returns output + grads."""
        from paddle.distributed.fleet.utils import recompute

        query, key, w_v, x, qr = _make_inputs(
            SEQLEN, seed=seed, with_hidden=True
        )
        module.train()
        module.clear_gradients()
        q = _leaf(query)

        def fn(qin):
            return module(
                qin, key, None, None, row_end, v_b_proj_weight=w_v, x=x, qr=qr
            )

        out = recompute(fn, q) if use_recompute else fn(q)
        out.cast("float32").sum().backward()
        grads = {
            name: (None if p.grad is None else p.grad.detach().cast("float32"))
            for name, p in module.named_parameters()
        }
        grads["__query__"] = (
            None if q.grad is None else q.grad.detach().cast("float32")
        )
        return out.detach().cast("float32"), grads

    def _equivalence(self, layout):
        row_end = _row_end(layout, SEQLEN)
        expected = _full_causal_table(layout, SEQLEN)

        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        out_off, g_off = self._run(self.module, row_end, use_recompute=False)
        idx_off = _CAPTURED[-1]
        loss_off = _tracker_value(self.module.layer_number - 1)
        n_calls_off = len(_CAPTURED)

        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        out_on, g_on = self._run(self.module, row_end, use_recompute=True)
        loss_on = _tracker_value(self.module.layer_number - 1)

        # The recompute forward really ran (otherwise the rest proves nothing).
        self.assertEqual(n_calls_off, 1)
        self.assertGreaterEqual(
            len(_CAPTURED), 2, "recompute did not re-forward the layer"
        )
        # Bit-identical index tables: across the two recompute passes, against
        # the no-recompute run, and against the analytic table. No tolerance --
        # the warmup table has no floating-point scoring in it at all.
        for cap in _CAPTURED:
            np.testing.assert_array_equal(cap, idx_off)
            np.testing.assert_array_equal(cap, expected)
        _check_index_invariants(
            self, idx_off, row_end, SEQLEN, expect_full=True
        )

        out_rel = _rel(out_on, out_off)
        self.assertEqual(set(g_on), set(g_off))
        grad_rels = {name: _grad_rel(g_on[name], g_off[name]) for name in g_off}
        worst = max(grad_rels, key=grad_rels.get)
        print(
            f"[warmup recompute {layout}] out_rel={out_rel:.3e} "
            f"worst_grad={worst}:{grad_rels[worst]:.3e} "
            f"loss_off={loss_off!r} loss_on={loss_on!r} "
            f"all_grads={ {k: f'{v:.2e}' for k, v in grad_rels.items()} }"
        )
        # Same tolerances as the phase-3 equivalence test
        # (test_hybrid_mla_recompute_mtp_ckpt.TestRecomputeEquivalence).
        self.assertLess(out_rel, 1e-5, f"{layout} output rel={out_rel}")
        for name, rel in grad_rels.items():
            self.assertLess(rel, 5e-3, f"{layout} grad[{name}] rel={rel}")
        # The indexer loss is attached on the grad-enabled pass only, so it is
        # counted once -- not twice, and not lost.
        self.assertIsNotNone(loss_off)
        self.assertIsNotNone(loss_on)
        self.assertGreater(loss_off, 0.0)
        self.assertAlmostEqual(
            loss_on / loss_off,
            1.0,
            delta=1e-3,
            msg=f"indexer loss counted {loss_on / loss_off:.3f}x under recompute",
        )

    def test_recompute_equivalence_two_documents(self):
        self._equivalence(TWO_DOCS)

    def test_recompute_equivalence_single_document(self):
        """The layout phase 3 cannot assert on: warmup is exact here."""
        self._equivalence(ONE_DOC)

    def _indexer_call_count(self, use_sparse_loss):
        """Indexer selector calls across both passes, per backend.

        Two selectors have to be counted separately now: phase 3 selects with
        the **cuDNN** top-k kernel, phase 2 with the **tilelang** one at
        ``topk_effective = s_global`` (its full-candidate mode). Counting only
        cuDNN would report "zero top-k calls" for warmup and hide the one call
        it does make.
        """
        import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as fwd_mod
        import paddlefleet.tilelang_ops as tl_mod

        config = _create_mqa_config("mqa_dsa", loss_coeff=0.01)
        config.dsa_indexer_use_sparse_loss = use_sparse_loss
        module = _build_module(config, bf16=True)
        self.assertEqual(module.indexer_use_sparse_loss, use_sparse_loss)

        cudnn_calls = []
        tl_calls = []
        before_calls = []
        inner_topk = fwd_mod.cudnn_indexer_topk_fwd
        inner_tl = tl_mod.csa_indexer_topk_fwd
        inner_before = module.indexer.forward_before_topk

        def rec_topk(*args, **kwargs):
            cudnn_calls.append(int(kwargs["topk_effective"]))
            return inner_topk(*args, **kwargs)

        def rec_tl(*args, **kwargs):
            tl_calls.append(int(kwargs["topk_effective"]))
            return inner_tl(*args, **kwargs)

        def rec_before(*args, **kwargs):
            before_calls.append(1)
            return inner_before(*args, **kwargs)

        fwd_mod.cudnn_indexer_topk_fwd = rec_topk
        tl_mod.csa_indexer_topk_fwd = rec_tl
        module.indexer.forward_before_topk = rec_before
        _CAPTURED.clear()
        try:
            self._run(module, _row_end(TWO_DOCS, SEQLEN), use_recompute=True)
        finally:
            fwd_mod.cudnn_indexer_topk_fwd = inner_topk
            tl_mod.csa_indexer_topk_fwd = inner_tl
            module.indexer.forward_before_topk = inner_before
        return len(before_calls), cudnn_calls, tl_calls, len(_CAPTURED)

    def test_no_grad_pass_skips_the_indexer_only_in_warmup(self):
        """The warmup early exit, observed through the recompute double pass.

        Under recompute the layer is forwarded twice: once under ``no_grad``
        (no loss needed) and once grad-enabled. In warmup the first pass takes
        the early exit, so the indexer projections run exactly *once* even though
        attention was built twice, and the **cuDNN** top-k kernel -- phase 3's
        selector -- runs **zero** times: warmup reads no ``index_topk``. What it
        does run is one **tilelang** call at ``topk_effective == SEQLEN``, its
        full-candidate mode, and exactly one, on the grad-enabled pass only.
        Phase 3 has no such exit: attention consumes the ranking, so the
        projections and the cuDNN kernel both run on both passes. The contrast is
        the discriminator.
        """
        n_before_w, cudnn_w, tl_w, n_attn_w = self._indexer_call_count(False)
        n_before_s, cudnn_s, tl_s, n_attn_s = self._indexer_call_count(True)
        print(
            f"[warmup indexer calls] warmup before_topk={n_before_w} "
            f"cudnn={cudnn_w} tilelang={tl_w} attn_forwards={n_attn_w} || "
            f"phase3 before_topk={n_before_s} cudnn={cudnn_s} "
            f"tilelang={tl_s} attn_forwards={n_attn_s}"
        )
        self.assertGreaterEqual(n_attn_w, 2, "recompute did not re-forward")
        self.assertGreaterEqual(n_attn_s, 2, "recompute did not re-forward")
        self.assertEqual(n_before_w, 1)
        self.assertEqual(cudnn_w, [], "warmup called the cuDNN top-k kernel")
        # One tilelang call, on the grad-enabled pass only, over every column.
        self.assertEqual(tl_w, [SEQLEN])
        # Same wrapping, sparse phase: no early exit, both passes pay for it.
        self.assertEqual(n_before_s, 2)
        self.assertEqual(len(cudnn_s), 2)
        self.assertEqual(cudnn_s, [INDEX_TOPK, INDEX_TOPK])
        self.assertEqual(tl_s, [], "phase 3 selected with the tilelang kernel")


def _mtp_config(use_sparse_loss, loss_coeff=0.01):
    """Production MTP shape: 43 backbone layers + 1 next-n predict layer."""
    config = _create_mqa_config(
        "mqa_dsa", loss_coeff=loss_coeff, num_hidden_layers=43
    )
    config.dsa_indexer_use_sparse_loss = use_sparse_loss
    config.num_nextn_predict_layers = 1
    config.pad_token_id = 0
    return config


@_GPU
class TestWarmupMTP(unittest.TestCase):
    """The warmup shape inside the MTP layer.

    ``MultiTokenPredictionLayer.__init__`` builds its ``transformer_layer``
    without passing ``pg_collection`` (``multi_token_prediction.py:419-423``), so
    the MTP ``-2`` layer is an ordinary ``MQALatentAttention`` reading the same
    config -- there is no MTP-specific branch that could keep it on the phase-3
    shape. These tests pin that: same construction as
    ``test_hybrid_mla_mtp_layer43_w6._build_mtp_module`` but with the phase-2
    switch off.
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
        DSAIndexerLossLoggingHelper.num_layers = None

    @staticmethod
    def _build(use_sparse_loss=False, loss_coeff=0.01):
        module = _build_module(
            _mtp_config(use_sparse_loss, loss_coeff),
            layer_number=0,
            bf16=True,
            is_mtp=True,
        )
        assert module.layer_number == 0
        return module

    def _call(self, module, row_end, training=True):
        query, key, w_v, x, qr = _make_inputs(SEQLEN, seed=5, with_hidden=True)
        module.train() if training else module.eval()
        return module(
            query, key, None, None, row_end, v_b_proj_weight=w_v, x=x, qr=qr
        )

    def test_mtp_attention_is_full_causal_in_warmup(self):
        """The MTP ``-2`` layer's attention takes the full causal table, and its
        output is bit-identical to the indexer-less ``mqa_full_causal`` MTP
        layer -- the same equality the backbone layer has."""
        row_end = _row_end(TWO_DOCS, SEQLEN)
        module = self._build()
        self.assertFalse(module.indexer_use_sparse_loss)

        reference_cfg = _create_mqa_config("mqa", num_hidden_layers=43)
        reference_cfg.num_nextn_predict_layers = 1
        reference = _build_module(
            reference_cfg, layer_number=0, bf16=True, is_mtp=True
        )
        self.assertIsNone(reference.indexer)

        _CAPTURED.clear()
        out_ref = _fp32(self._call(reference, row_end, training=False))
        table_ref = _CAPTURED[-1]

        _CAPTURED.clear()
        out_warm = _fp32(self._call(module, row_end, training=True))
        table_warm = _CAPTURED[-1]

        expected = _full_causal_table(TWO_DOCS, SEQLEN)
        np.testing.assert_array_equal(table_warm, expected)
        np.testing.assert_array_equal(table_warm, table_ref)
        _check_index_invariants(
            self, table_warm, row_end, SEQLEN, expect_full=True
        )
        maxabs = float(np.max(np.abs(out_warm - out_ref)))
        print(f"[warmup mtp] out maxabs vs mqa_full_causal MTP = {maxabs!r}")
        np.testing.assert_array_equal(out_warm, out_ref)

    def test_mtp_indexer_loss_denominator_counts_the_mtp_minus2_layer(self):
        """``get_total_num_layers`` is 44 and the tracker row is the MTP one.

        The MTP layer keeps ``layer_number=0``, so ``save_loss_to_tracker``
        writes ``values[-1]``: a pre-existing logging blemish already pinned by
        ``test_hybrid_mla_recompute_mtp_ckpt.TestMTPTrackerSlot``. Asserted here
        as-is, deliberately not "fixed" -- what this test adds is that the
        warmup phase still reaches the tracker at all.
        """
        module = self._build()
        self.assertEqual(
            DSAIndexerLossLoggingHelper.get_total_num_layers(module.config), 44
        )
        out = self._call(module, _row_end(TWO_DOCS, SEQLEN), training=True)
        out.cast("float32").sum().backward()
        values = DSAIndexerLossLoggingHelper.tracker["values"]
        self.assertEqual(list(values.shape), [44])
        nonzero = [i for i, v in enumerate(values.numpy()) if v != 0.0]
        loss = float(values.numpy()[-1])
        print(f"[warmup mtp tracker] nonzero_slots={nonzero} values[-1]={loss}")
        self.assertEqual(nonzero, [43])
        self.assertGreater(loss, 0.0)

    def test_track_indexer_metrics_denominator_over_the_enum(self):
        """The cross-repo API now takes the enum string, and only ``mqa_dsa``
        adds the ``-2`` layers to the denominator.

        ``_prod_csa_ratios`` has no CSA layer (``1 < ratio < 128``) at all, so
        the ``-2`` contribution is the whole denominator: ``mqa_dsa`` gives 6
        (five backbone + the MTP layer), the other two modes give 0, which is
        the "no indexer anywhere" path that clears the tracker and reports
        nothing.
        """
        ratios = _prod_csa_ratios()
        total = 0.0
        for mode, expect in (
            ("mqa_dsa", 6),
            ("mqa_full_causal", None),
            ("mha", None),
        ):
            with self.subTest(mode=mode):
                DSAIndexerLossLoggingHelper.tracker.clear()
                values = paddle.zeros([44])
                for slot in _PROD_MINUS2:
                    values[slot] = 1.0
                DSAIndexerLossLoggingHelper.tracker["values"] = values
                total = float(values.sum())
                sink = {}
                DSAIndexerLossLoggingHelper.track_indexer_metrics(
                    loss_scale=1.0,
                    iteration=0,
                    total_loss_dict=sink,
                    num_layers=44,
                    csa_compress_ratios=ratios,
                    hybrid_mla_attention=mode,
                )
                got = sink.get("indexer loss")
                got = None if got is None else float(got)
                print(f"[track_indexer_metrics {mode}] sum={total} avg={got}")
                if expect is None:
                    self.assertIsNone(got)
                else:
                    self.assertEqual(len(_PROD_MINUS2), expect)
                    self.assertAlmostEqual(got, total / expect, places=6)
        self.assertEqual(total, 6.0)


def ref_rope_interleaved(x, pos, base):
    """GPT-J / interleaved layout: pair channel 2j with 2j+1 in place."""
    d = x.shape[-1]
    ang = ref_angles(np.array([pos]), d, base)[0]
    cos, sin = np.cos(ang), np.sin(ang)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x, dtype=np.float64)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = b * cos + a * sin
    return out


class _WeightExposingLinear(BiasedLinear):
    """``BiasedLinear`` plus the ``.weight`` the absorption reads off
    ``kv_b_proj``."""

    @property
    def weight(self):
        return self.linear.weight


_MLA_SPEC = MLASelfAttentionSublayersSpec(
    q_proj=BiasedLinear,
    q_a_proj=BiasedLinear,
    q_b_proj=BiasedLinear,
    kv_a_proj_with_mqa=BiasedLinear,
    kv_b_proj=_WeightExposingLinear,
    core_attention=DotProductAttention,
    o_proj=BiasedLinear,
    q_a_layernorm=LayerNormStub,
    kv_a_layernorm=LayerNormStub,
)


@_GPU
class TestWarmupRope(unittest.TestCase):
    """The phase-2 switch must not reach RoPE, anywhere.

    Two independent RoPE users live on a ``-2`` layer: ``MLASelfAttention``
    rotates q/k *before* dispatching to the ``mqa_latent`` branch, and the DSA
    indexer keeps its own plain RoPE. The switch changes neither; every
    assertion below is against the fp64 reference of
    ``test_hybrid_mla_rope_audit``, never against the implementation.
    """

    def _mla(self, mode, use_sparse_loss=True):
        config = _create_mqa_config(mode)
        config.dsa_indexer_use_sparse_loss = use_sparse_loss
        paddle.seed(123)
        return MLASelfAttention(
            config=config, sublayers_spec=_MLA_SPEC, layer_number=1
        )

    def test_main_attention_rope_is_untouched_by_the_phase_switch(self):
        """warmup q/k == phase-3 q/k == the ``mha`` rope sub-blocks, exactly.

        ``get_query_key_value_tensors`` is where the rotation happens. Sharing
        one ``state_dict`` across the three modes (the key sets are identical --
        that is the whole point of activation-level absorption) makes the
        comparison meaningful, and the result must be bit-equality, since the
        switch is not read on this code path at all.
        """
        mha = self._mla("mha")
        warm = self._mla("mqa_dsa", use_sparse_loss=False)
        ph3 = self._mla("mqa_dsa", use_sparse_loss=True)
        self.assertEqual(set(mha.state_dict()), set(warm.state_dict()))
        self.assertEqual(set(mha.state_dict()), set(ph3.state_dict()))
        state = mha.state_dict()
        warm.set_state_dict(state)
        ph3.set_state_dict(state)
        self.assertFalse(mha.mqa_latent)
        self.assertTrue(warm.mqa_latent)

        paddle.seed(7)
        hidden = paddle.randn([1, 64, HIDDEN]) * 0.5
        out = {}
        for name, module in (("mha", mha), ("warm", warm), ("ph3", ph3)):
            module.eval()
            query, key = module.get_query_key_value_tensors(hidden)[:2]
            out[name] = (query, key)

        rope_dim = mha.config.hybrid_mla_qk_rope_head_dim
        pairs = {
            "warm_vs_ph3_q": (out["warm"][0], out["ph3"][0]),
            "warm_vs_ph3_k": (out["warm"][1], out["ph3"][1]),
            # The latent q/k carry the rope block in their trailing dims; the
            # nope halves differ in shape between mha and latent, the rope
            # sub-block does not. ``mha`` keeps K per head, the latent path
            # keeps the single shared head, so take head 0 on both.
            "mha_vs_warm_q_pe": (
                out["mha"][0][..., -rope_dim:],
                out["warm"][0][..., -rope_dim:],
            ),
            "mha_vs_warm_k_pe": (
                out["mha"][1][:, :, :1, -rope_dim:],
                out["warm"][1][:, :, :1, -rope_dim:],
            ),
        }
        measured = {}
        for name, (a, b) in pairs.items():
            measured[name] = float((a - b).abs().max())
        print(f"[warmup rope main] rope_dim={rope_dim} maxabs={measured}")
        for name, (a, b) in pairs.items():
            np.testing.assert_array_equal(
                a.numpy(), b.numpy(), err_msg=f"{name} moved"
            )

    def test_indexer_rope_is_plain_and_correct_in_warmup(self):
        """The indexer's own RoPE, in warmup, against the fp64 reference.

        Three things at once: the frequency table is plain RoPE (base 10000, not
        the compressed layers' YaRN / ``csa_compress_rotary_base``), the
        ``dsa_indexer_rotary_interleaved`` layout switch is live (each setting
        matches its own reference and *not* the other one -- the cross error is
        the self-calibration), and the ``nope`` tail is left bit-untouched.
        """
        seqlen = 16
        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                config = _warmup_config()
                config.dsa_indexer_rotary_interleaved = interleaved
                paddle.seed(99)
                indexer = _build_module(config, bf16=False).indexer
                self.assertIsNotNone(indexer)

                inv = np.asarray(
                    indexer.rotary_pos_emb.inv_freq.astype("float64").numpy()
                )
                inv_err = float(
                    np.max(
                        np.abs(
                            inv
                            - ref_inv_freq(indexer.rope_head_dim, ROPE_THETA)
                        )
                    )
                )

                rng = np.random.default_rng(3)
                # fp32 round-tripped up front, so "untouched" can be asserted
                # bit-exactly rather than against an fp64 draw the kernel never
                # saw.
                x = (
                    rng.standard_normal((1, seqlen, 1, indexer.head_dim))
                    .astype("float32")
                    .astype("float64")
                )
                freqs = indexer.rotary_pos_emb(seqlen, packed_seq=False)
                got = np.asarray(
                    indexer._apply_rope(
                        paddle.to_tensor(x.astype("float32")), freqs, 1.0
                    ).numpy(),
                    dtype=np.float64,
                )
                rope_dim = indexer.rope_head_dim
                nope_err = float(
                    np.max(np.abs(got[..., rope_dim:] - x[..., rope_dim:]))
                )
                pe = x[..., :rope_dim]
                own = (
                    ref_rope_interleaved if interleaved else ref_rope_halfsplit
                )
                other = (
                    ref_rope_halfsplit if interleaved else ref_rope_interleaved
                )
                own_err = max(
                    float(
                        np.max(
                            np.abs(
                                got[0, p, 0, :rope_dim]
                                - own(pe[0, p, 0], p, ROPE_THETA)
                            )
                        )
                    )
                    for p in range(seqlen)
                )
                other_err = max(
                    float(
                        np.max(
                            np.abs(
                                got[0, p, 0, :rope_dim]
                                - other(pe[0, p, 0], p, ROPE_THETA)
                            )
                        )
                    )
                    for p in range(seqlen)
                )
                print(
                    f"[warmup rope indexer interleaved={interleaved}] "
                    f"inv_freq_err={inv_err:.3e} nope_untouched={nope_err!r} "
                    f"own_layout_err={own_err:.3e} "
                    f"other_layout_err={other_err:.3e}"
                )
                self.assertLess(inv_err, 1e-6, "not plain rope base 10000")
                self.assertEqual(nope_err, 0.0, "nope half was rotated")
                self.assertLess(own_err, 1e-5)
                self.assertGreater(other_err, 1e-2, "layout switch is not live")

    def test_indexer_rope_output_matches_phase_three_bitwise(self):
        """``forward_before_topk`` q/k are identical across the two phases.

        The indexer still runs in warmup; only the *consumption* of its ranking
        changes. Copying one ``state_dict`` across, the pre-top-k activations --
        which is where all the RoPE is -- must be bit-equal.
        """
        seqlen = 64
        warm = _build_module(_warmup_config(), bf16=True)
        ph3 = _build_module(
            _create_mqa_config("mqa_dsa", loss_coeff=0.01), bf16=True
        )
        self.assertTrue(ph3.indexer_use_sparse_loss)
        ph3.set_state_dict(warm.state_dict())

        paddle.seed(11)
        x = (paddle.randn([1, seqlen, HIDDEN]) * 0.5).cast("bfloat16")
        qr = (paddle.randn([1, seqlen, Q_LORA]) * 0.5).cast("bfloat16")
        outs = []
        for module in (warm, ph3):
            module.eval()
            with paddle.no_grad():
                outs.append(module.indexer.forward_before_topk(x, qr))
        measured = []
        for a, b in zip(outs[0], outs[1]):
            if not isinstance(a, paddle.Tensor):
                self.assertEqual(a, b)
                continue
            measured.append(float((a - b).cast("float32").abs().max()))
            np.testing.assert_array_equal(
                a.cast("float32").numpy(), b.cast("float32").numpy()
            )
        print(f"[warmup rope indexer vs phase3] maxabs={measured}")
        self.assertTrue(measured)

    def test_apply_rope_fusion_stays_excluded_from_latent_mqa(self):
        """Construction-time exclusion (``multi_latent_attention.py``): the fused
        kernel writes the per-head K/V that absorption skips.

        Negative case for both latent modes, including the warmup pairing, plus
        the ``mha`` positive control that proves the exclusion is scoped to
        latent MQA and not simply "fusion is broken here".
        """
        for mode, sparse, should_raise in (
            ("mha", True, False),
            ("mqa", True, True),
            ("mqa_dsa", True, True),
            ("mqa_dsa", False, True),
        ):
            with self.subTest(mode=mode, use_sparse_loss=sparse):
                config = _create_mqa_config(mode)
                config.dsa_indexer_use_sparse_loss = sparse
                config.apply_rope_fusion = True
                if should_raise:
                    with self.assertRaises(ValueError) as ctx:
                        MLASelfAttention(
                            config=config,
                            sublayers_spec=_MLA_SPEC,
                            layer_number=1,
                        )
                    self.assertIn("apply_rope_fusion", str(ctx.exception))
                else:
                    module = MLASelfAttention(
                        config=config,
                        sublayers_spec=_MLA_SPEC,
                        layer_number=1,
                    )
                    self.assertFalse(module.mqa_latent)


@_GPU
class TestRecomputeInnerForwardBitIdentical(unittest.TestCase):
    """The forward *inside* backward must equal the forward outside it, bitwise.

    ``TestWarmupRecompute`` above compares recompute-on against recompute-off and
    compares the two index tables, which pins the sparsity pattern. It does not
    compare the two forwards' *outputs*, because paddle discards the first one --
    so a divergence in R2's activations that happened to leave the column set
    alone would go unnoticed and be differentiated silently.

    Here both returns are kept and compared. ``maxabs == 0.0`` is the expected
    result, not a tolerance: the recomputed forward re-executes the same kernels
    on the same saved inputs, and the index table is an integer function of the
    document bounds. The captured-call count is asserted first, otherwise a
    single-forward implementation would make the comparison vacuous.
    """

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

    # ``mqa`` -> "mqa_full_causal"; the two ``mqa_dsa`` rows are warmup (wide
    # loss, full-causal attention) and phase 3 (narrow loss, top-k attention).
    _MODES = (("mqa_dsa", False), ("mqa_dsa", True), ("mqa", None))
    # 256 saturates window+topk, 512 does not -- see the module docstring of
    # ``test_hybrid_mla_warmup_doc_mask_loss``.
    _SHAPES = ((SEQLEN, TWO_DOCS), (512, [200, 312]))

    def _module(self, mode, sparse_loss):
        config = _create_mqa_config(mode, loss_coeff=0.01)
        if sparse_loss is not None:
            config.dsa_indexer_use_sparse_loss = sparse_loss
        config.pad_token_id = 0
        return _build_module(config, bf16=True)

    def _capture_two_forwards(self, module, seqlen, layout):
        """Run one recomputed train step; return both forwards' outputs."""
        import types

        from paddle.distributed.fleet.utils import recompute

        query, key, w_v, x, qr = _make_inputs(seqlen, seed=7, with_hidden=True)
        row_end = _row_end(layout, seqlen)
        module.train()
        module.clear_gradients()
        q = _leaf(query)

        outs = []
        real = type(module).forward

        def spy(zelf, *args, **kwargs):
            result = real(zelf, *args, **kwargs)
            tensor = result[0] if isinstance(result, tuple) else result
            outs.append(tensor.detach().cast("float32").numpy().copy())
            return result

        module.forward = types.MethodType(spy, module)
        try:
            out = recompute(
                lambda qin: module(
                    qin,
                    key,
                    None,
                    None,
                    row_end,
                    v_b_proj_weight=w_v,
                    x=x,
                    qr=qr,
                    input_ids=paddle.ones([1, seqlen], dtype="int64"),
                ),
                q,
            )
            out.cast("float32").sum().backward()
        finally:
            del module.forward
        return outs

    def test_inner_forward_output_is_bit_identical(self):
        for mode, sparse_loss in self._MODES:
            for seqlen, layout in self._SHAPES:
                with self.subTest(mode=mode, sparse=sparse_loss, s=seqlen):
                    _CAPTURED.clear()
                    DSAIndexerLossLoggingHelper.tracker.clear()
                    module = self._module(mode, sparse_loss)
                    outs = self._capture_two_forwards(module, seqlen, layout)
                    self.assertGreaterEqual(
                        len(outs),
                        2,
                        "recompute did not run a second forward, so this "
                        "comparison would be vacuous",
                    )
                    self.assertEqual(
                        float(np.abs(outs[0] - outs[1]).max()),
                        0.0,
                        "the forward inside backward differs from the outer "
                        "forward",
                    )

    def test_inner_forward_index_table_is_bit_identical(self):
        for mode, sparse_loss in self._MODES:
            for seqlen, layout in self._SHAPES:
                with self.subTest(mode=mode, sparse=sparse_loss, s=seqlen):
                    _CAPTURED.clear()
                    DSAIndexerLossLoggingHelper.tracker.clear()
                    module = self._module(mode, sparse_loss)
                    self._capture_two_forwards(module, seqlen, layout)
                    self.assertGreaterEqual(
                        len(_CAPTURED), 2, "only one sparse-kernel call"
                    )
                    first, second = _CAPTURED[0], _CAPTURED[-1]
                    self.assertEqual(
                        int((first != second).sum()),
                        0,
                        "the recomputed forward selected different columns",
                    )


if __name__ == "__main__":
    unittest.main()
