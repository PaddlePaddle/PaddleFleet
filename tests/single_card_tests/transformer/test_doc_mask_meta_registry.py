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

"""CPU-only tests for the entire document-mask metadata stack.

One file, nothing below actually runs a kernel:

* :class:`TestDocMetaRegistry` -- the registry slot lifecycle / audit /
  geometry asserts / the CSA_MQA_RATIO warm skip;
* :class:`TestDocMaskMetaConfigValidation` -- ``TransformerConfig`` startup
  guards for the sharing switches;
* :class:`TestMQADocMetaDerivedTables` -- the pure MQADocMeta index tables
  (window / valid-range / cu_seqlens / CP slicing);
* :class:`TestDocMaskLayerWiring` -- ``register`` / ``advance`` /
  ``_docmask_meta_kwargs`` on a minimal ``HyperConnectionTransformerLayer``;
* :class:`TestMQASlotLookup` -- the forward's slot lookup with the sparse
  phase forward stubbed (kernel-free, runs on any machine);
* :class:`TestMultiLatentForwardInferenceKwargs` -- the base MLA template
  forward's inference-kwargs extraction (``multi_latent_attention.py:950-957``)
  reaches ``is_decode`` on CPU, backed by a white-box stub module;
* :class:`TestEmbeddingInitBranch` -- ``TransformerConfig``'s embedding init
  method resolution (``transformer_config.py:1762-1784``) and the
  ``hybrid_mla_attention`` ratio-counts loop.

The kernel-backed end-to-end equivalence stays in
``test_doc_mask_meta_cache_equivalence.py`` (GPU-gated).
"""

import unittest
import unittest.mock
from types import SimpleNamespace

import numpy as np
import paddle

from paddlefleet.transformer.doc_mask_meta_registry import (
    doc_mask_meta_registry,
)
from paddlefleet.transformer.dsv4_hybrid_attention import (
    DSv4HybridAttention,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.mqa_latent_attention import (
    MQADocMeta,
    MQALatentAttention,
)
from paddlefleet.transformer.multi_latent_attention import MultiLatentAttention
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    IdentityFuncOp,
    TransformerLayerSublayersSpec,
)
from paddlefleet.utils import init_method_normal

from .hybrid_mla_utils import (
    WINDOW,
    _build_module,
    _create_mqa_config,
    _FakePGCollection,
    _make_inputs,
    _pad_row_end,
    _row_end,
)

_S = 16
_W = 3
_SEQ = 64
_ACC = 2


def _csa_mask(doc_lens, seqlen):
    """``[1, 1, s, 1]`` int32 exclusive document-end mask (what
    ``CSADocMaskMetadata`` reads)."""
    return _row_end(doc_lens, seqlen)


class TestDocMaskMetaRegistry(unittest.TestCase):
    """The registry is a process-global singleton: every test normalises it
    with ``begin_step`` first and never relies on a previous test's state."""

    def setUp(self):
        self.reg = doc_mask_meta_registry
        self.reg.begin_step(2)
        self.row_end = _row_end([7, 9], 16)
        self.csa_mask = _row_end([7, 9], 16)

    # ------------------------------------------------------------------
    # step lifecycle
    # ------------------------------------------------------------------
    def test_begin_step_clears_store_and_resets_counters(self):
        self.reg.begin_step(3)
        self.reg.preload_mqa(0, 1, 16, self.row_end, ("main",), 3)
        stats = self.reg.begin_step(2)
        self.assertEqual(set(stats), {"prebuild", "hit"})
        # the slot from the previous step is gone
        self.assertIsNone(self.reg.get_mqa(0, 1, 16, ("main",)))

    # ------------------------------------------------------------------
    # MQA slots
    # ------------------------------------------------------------------
    def test_preload_mqa_then_get_is_same_object(self):
        self.reg.preload_mqa(1, 1, 16, self.row_end, ("main",), 3)
        meta = self.reg.get_mqa(1, 1, 16, ("main",))
        self.assertIsNotNone(meta)
        self.assertEqual((meta.batch_size, meta.seqlen), (1, 16))

    def test_get_mqa_absent_group_returns_none(self):
        # ("mtp", 0) is never prebuilt by the trainer: miss -> None -> the
        # layer builds privately. Requires the plain int mb_idx same as prod.
        self.reg.preload_mqa(0, 1, 16, self.row_end, ("main",), 3)
        for group in (("mtp", 0), ("mtp", 1), ("other",)):
            self.assertIsNone(self.reg.get_mqa(0, 1, 16, group))

    def test_preload_mqa_requires_begin_step(self):
        # no begin_step: store is empty, slot is missing, but per contract a
        # main-group _mqa_ miss returns None (fallback path) not an error.
        self.reg.begin_step(1)  # re-normalise; then never preload
        self.assertIsNone(self.reg.get_mqa(0, 1, 16, ("main",)))

    def test_get_mqa_batch_mismatch_raises(self):
        self.reg.preload_mqa(0, 1, 16, self.row_end, ("main",), 3)
        with self.assertRaises(ValueError):
            self.reg.get_mqa(0, 2, 16, ("main",))

    def test_get_mqa_seqlen_mismatch_raises(self):
        self.reg.preload_mqa(0, 1, 16, self.row_end, ("main",), 3)
        with self.assertRaises(ValueError):
            self.reg.get_mqa(0, 1, 32, ("main",))

    # ------------------------------------------------------------------
    # CSA slots
    # ------------------------------------------------------------------
    def test_preload_csa_get_hit_is_same_object(self):
        self.reg.preload(
            1,
            4,
            1,
            16,
            self.csa_mask,
            dense_mode=False,
            mask_group=("main",),
            window_size=3,
        )
        meta = self.reg.get(1, 4, 1, 16, ("main",))
        self.assertIsNotNone(meta)
        self.assertEqual(meta.ratio, 4)

    def test_preload_ratio_minus1_warm_short_circuits(self):
        # doc_mask_meta_registry._warm (line 178): the full-causal MQA ratio -1
        # deliberately skips the linear index tables, so preload must accept it
        # without touching them.
        self.reg.preload(
            0,
            -1,
            1,
            16,
            self.csa_mask,
            dense_mode=False,
            mask_group=("main",),
            window_size=3,
        )
        self.assertIsNotNone(self.reg.get(0, -1, 1, 16, ("main",)))

    def test_get_ratio_normalises_like_preload(self):
        # preload stores under max(1, ratio); get must normalise the same way,
        # so a raw 0/-2 ratio from the config still hits.
        self.reg.preload(
            0,
            -2,
            1,
            16,
            self.csa_mask,
            dense_mode=False,
            mask_group=("main",),
            window_size=3,
        )
        self.assertIsNotNone(self.reg.get(0, 1, 1, 16, ("main",)))

    def test_get_main_group_miss_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.reg.get(0, 4, 1, 16, ("main",))

    def test_get_mtp_group_miss_returns_none(self):
        # absent-by-design group: the layer falls back to building privately
        self.assertIsNone(self.reg.get(0, 4, 1, 16, ("mtp", 0)))

    def test_get_slot_geometry_mismatch_raises(self):
        self.reg.preload(
            1,
            4,
            1,
            16,
            self.csa_mask,
            dense_mode=False,
            mask_group=("main",),
            window_size=3,
        )
        with self.assertRaises(ValueError):
            self.reg.get(1, 4, 2, 16, ("main",))

    # ------------------------------------------------------------------
    # forward counter (advance / audit)
    # ------------------------------------------------------------------
    def test_advance_yields_slots_then_wraps(self):
        self.reg.begin_step(2)
        got = [self.reg.advance(("adv", 1), True) for _ in range(4)]
        self.assertEqual(got, [0, 1, 0, 1])

    def test_advance_training_inside_nograd_raises(self):
        # recompute wrapper: no-grad training forward must fail loudly instead
        # of silently reading the previous micro-batch's slot
        self.reg.begin_step(2)
        self.reg.advance(("adv", 1), True)  # warm the tracer
        with paddle.no_grad(), self.assertRaises(RuntimeError):
            self.reg.advance(("adv", 1), True)

    def test_advance_evaluation_inside_nograd_ok(self):
        # inference advances normally (no backward, so no replay hazard)
        self.reg.begin_step(2)
        with paddle.no_grad():
            self.assertEqual(self.reg.advance(("adv", 1), False), 0)

    def test_advance_unregistered_key_starts_at_zero(self):
        # a consumer that never called register() still gets a sane counter
        self.reg.begin_step(3)
        self.assertEqual(self.reg.advance(("late", 0), True), 0)

    def test_check_accepts_unrun_and_exact_run(self):
        self.reg.begin_step(2)
        key1, key2 = ("a", 0), ("b", 1)
        self.reg.register(key1)
        self.reg.register(key2)
        self.reg.advance(key1, True)  # cnt 0
        self.reg.advance(key1, True)  # cnt 1 == acc-1
        # key2 never ran: -1 is allowed (elastic / mtp-disabled layers)
        self.reg.check()

    def test_check_rejects_wrong_run_count(self):
        self.reg.begin_step(2)
        key = ("c", 2)
        self.reg.register(key)
        self.reg.advance(key, True)
        self.reg.advance(key, True)
        self.reg.advance(key, True)  # cnt 2 != acc-1
        with self.assertRaises(RuntimeError):
            self.reg.check()

    def test_check_after_preload_get_happy_path(self):
        # the producer->consumer->audit cycle exactly as the trainer runs it
        self.reg.begin_step(2)
        for mb in range(2):
            self.reg.preload_mqa(mb, 1, 16, self.row_end, ("main",), 3)
        for mb in range(2):
            self.assertIsNotNone(self.reg.get_mqa(mb, 1, 16, ("main",)))
        self.reg.check()


class TestMQADocMetaDerivedTables(unittest.TestCase):
    def setUp(self):
        # "mqa" mask: two documents [7, 9] exactly tiling seqlen 16.
        self.seqlen = _S
        self.row_end = _row_end([7, 9], _S)
        self.meta = MQADocMeta.build(self.row_end, 1, _S)

    # ------------------------------------------------------------------
    # build / boundary fields
    # ------------------------------------------------------------------
    def test_build_row_end_none_is_one_document(self):
        meta = MQADocMeta.build(None, 1, _S)
        self.assertEqual(meta.batch_size, 1)
        self.assertEqual(meta.seqlen, _S)
        self.assertTrue(bool(meta.is_valid.all().item()))
        self.assertEqual(int(meta.doc_start_per_pos[0]), 0)
        self.assertEqual(int(meta.doc_len_per_pos[-1]), _S)

    def test_boundaries_follow_the_mask(self):
        ds = np.asarray([0] * 7 + [7] * 9)
        dl = np.asarray([7] * 7 + [9] * 9)
        np.testing.assert_array_equal(self.meta.doc_start_per_pos.numpy(), ds)
        np.testing.assert_array_equal(self.meta.doc_len_per_pos.numpy(), dl)
        self.assertTrue(bool(self.meta.is_valid.all().item()))
        np.testing.assert_array_equal(
            self.meta.doc_lens.numpy(), np.asarray([7, 9])
        )

    # ------------------------------------------------------------------
    # window table
    # ------------------------------------------------------------------
    def test_window_topk_left_aligned_then_padded(self):
        idx = self.meta.window_topk_idxs(_W, 0, _S).numpy()[0]
        # row 0: only token 0 is usable -> [0, -1, -1]
        np.testing.assert_array_equal(idx[0], [0, -1, -1])
        # row 2: tokens 0..2 (win_start = max(0, 0) = 0)
        np.testing.assert_array_equal(idx[2], [0, 1, 2])
        # row 6: tokens 4..6 (win_start = max(0, 4) = 4)
        np.testing.assert_array_equal(idx[6], [4, 5, 6])
        # row 7: second document starts at 7 -> [7, -1, -1]
        np.testing.assert_array_equal(idx[7], [7, -1, -1])
        # row 15: tokens 13..15
        np.testing.assert_array_equal(idx[15], [13, 14, 15])

    def test_window_topk_padding_rows_are_all_minus_one(self):
        # _pad_row_end repeats the last document's end past token 7, so rows
        # 7..15 are invalid padding.
        meta = MQADocMeta.build(_pad_row_end([7], _S), 1, _S)
        idx = meta.window_topk_idxs(_W, 0, _S).numpy()[0]
        # rows past the doc content are invalid -> all -1
        for i in range(7, _S):
            np.testing.assert_array_equal(idx[i], [-1, -1, -1])
        # within the doc still the left-aligned window
        np.testing.assert_array_equal(idx[4], [2, 3, 4])

    def test_window_topk_cp_rows_are_sliced(self):
        full = self.meta.window_topk_idxs(_W, 0, _S).numpy()
        local = self.meta.window_topk_idxs(_W, 8, 8).numpy()
        self.assertEqual(local.shape, (1, 8, _W))
        np.testing.assert_array_equal(local[0], full[0][8:16])

    # ------------------------------------------------------------------
    # indexer valid range
    # ------------------------------------------------------------------
    def test_valid_range_no_window_is_causal_span(self):
        vr, empty = self.meta.indexer_valid_range(0, 0, _S)
        vr = vr.numpy()[0]
        # row 6: [doc_start(0), 7); row 7: [7, 8); row 15: [7, 16)
        np.testing.assert_array_equal(vr[6], [0, 7])
        np.testing.assert_array_equal(vr[7], [7, 8])
        np.testing.assert_array_equal(vr[15], [7, 16])
        # nothing is empty with window = 0
        self.assertFalse(bool(empty.sum().item()))

    def test_valid_range_window_clips_tail(self):
        vr, empty = self.meta.indexer_valid_range(_W, 0, _S)
        vr = vr.numpy()[0]
        empty = empty.numpy()[0]
        # row 3: causal avail 4, window 3 -> [0, 1)
        np.testing.assert_array_equal(vr[3], [0, 1])
        # row 6: avail 7 - 3 = 4 -> [0, 4)
        np.testing.assert_array_equal(vr[6], [0, 4])
        # rows 0..2 have their whole causal span swallowed by the window
        self.assertTrue(bool(empty[0]))
        self.assertFalse(bool(empty[3]))
        self.assertFalse(bool(empty[6]))
        np.testing.assert_array_equal(vr[15], [7, 13])

    def test_valid_range_padding_rows_reported_empty(self):
        # tail row 15 is pad (last doc's end repeated), so no causal budget:
        meta = MQADocMeta.build(_pad_row_end([0, 15], _S), 1, _S)
        self.assertEqual(int(meta.doc_lens.sum()), 15)
        vr, empty = meta.indexer_valid_range(_W, 0, _S)
        empty = empty.numpy()[0]
        self.assertTrue(bool(empty[15]))  # pad row
        self.assertTrue(bool(empty[2]))  # near doc start: window swallows span
        self.assertFalse(bool(empty[5]))

    def test_valid_range_cp_rows_are_sliced(self):
        vr_full, _ = self.meta.indexer_valid_range(0, 0, _S)
        vr_local, _ = self.meta.indexer_valid_range(0, 8, 8)
        np.testing.assert_array_equal(
            vr_local.numpy()[0], vr_full.numpy()[0][8:16]
        )

    # ------------------------------------------------------------------
    # cu_seqlens
    # ------------------------------------------------------------------
    def test_cu_seqlens_when_docs_tile_exactly(self):
        # the THD "cu_seqlens" arg is the per-document length list, exact
        # tiling required
        self.assertEqual(self.meta.cu_seqlens_arg(), [7, 9])

    def test_cu_seqlens_none_when_trailing_gap(self):
        # _pad_row_end repeats the last end, leaving a true non-tiling gap
        meta = MQADocMeta.build(_pad_row_end([7, 6], _S), 1, _S)
        self.assertEqual(int(meta.doc_lens.sum()), 13)
        self.assertIsNone(meta.cu_seqlens_arg())

    def test_cu_seqlens_none_row_end_none_single_doc_is_cached(self):
        meta = MQADocMeta.build(None, 1, _S)
        # one doc covering the whole sequence: tiles exactly
        self.assertEqual(meta.cu_seqlens_arg(), [_S])

    # ------------------------------------------------------------------
    # warm / immutability
    # ------------------------------------------------------------------
    def test_warm_prebuilds_every_consumer_table(self):
        meta = MQADocMeta.build(self.row_end, 1, _S)
        meta.warm(_W)  # must not raise; warmup(0) and warm(window) both built
        self.assertIsNotNone(meta._window_topk_idxs)
        self.assertTrue(meta._valid_range is not None)
        self.assertIn(0, meta._valid_range)
        self.assertIn(_W, meta._valid_range)
        self.assertIsNotNone(meta.cu_seqlens_arg())

    def test_tables_carry_no_grad(self):
        for table in (
            self.meta.window_topk_idxs(_W, 0, _S),
            self.meta.indexer_valid_range(0, 0, _S)[0],
        ):
            self.assertTrue(table.stop_gradient)


class TestDocMaskMetaConfigValidation(unittest.TestCase):
    def test_fused_mhc_requires_hyper_connections(self):
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=2,
                use_fused_mhc=True,
                enable_hyper_connections=False,
            )

    def test_invalid_hybrid_mla_attention_rejected(self):
        with self.assertRaises(ValueError):
            TransformerConfig(num_hidden_layers=2, hybrid_mla_attention="warp9")

    def test_mqa_mode_requires_dsv4_hybrid_variant(self):
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=2,
                hybrid_mla_attention="mqa_dsa",
                experimental_attention_variant="mla",
            )

    def test_mqa_mode_requires_minus2_ratio(self):
        # dsv4_hybrid but no -2 layer: no MLA layer can serve the mode
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=2,
                hybrid_mla_attention="mqa_full_causal",
                experimental_attention_variant="dsv4_hybrid",
                csa_compress_ratios=[4],
            )

    def test_mqa_dsa_warmup_and_sparse_loss_is_a_mixup(self):
        # train_indexer_only + sparse loss pick two different phase pairs:
        # reject, the two phases are fixed pairs by design
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=2,
                hybrid_mla_attention="mqa_dsa",
                experimental_attention_variant="dsv4_hybrid",
                csa_compress_ratios=[-2],
                train_indexer_only=True,
                dsa_indexer_use_sparse_loss=True,
            )

    # shared docmask switches: config.py:1893-1935 -------------------------
    def test_shared_switches_require_dsv4_variant(self):
        # 1894-1895: the switches only mean something on a DSv4-hybrid model.
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=2,
                csa_share_docmask_meta=True,
                mqa_share_docmask_meta=False,
            )

    def test_shared_switches_require_hyper_connections(self):
        # 1903-1904: only HyperConnectionTransformerLayer hands the slot down.
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=2,
                experimental_attention_variant="dsv4_hybrid",
                csa_compress_ratios=[4, 4],
                csa_share_docmask_meta=True,
                enable_hyper_connections=False,
            )

    def test_csa_share_requires_a_csa_kind_ratio(self):
        # 1920-1921: metadata sharing needs at least one -1/0/128/2-127 layer.
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=2,
                experimental_attention_variant="dsv4_hybrid",
                csa_compress_ratios=[-2, -2],
                csa_share_docmask_meta=True,
                enable_hyper_connections=True,
                hybrid_mla_attention="mqa_dsa",
                hybrid_mla_q_lora_rank=128,
                hybrid_mla_kv_lora_rank=512,
                hybrid_mla_qk_nope_head_dim=192,
                hybrid_mla_qk_rope_head_dim=64,
                hybrid_mla_v_head_dim=64,
                hybrid_mla_num_attention_heads=1,
                hybrid_mla_num_key_value_heads=1,
                dsa_index_n_heads=64,
                dsa_index_head_dim=128,
                dsa_index_topk=128,
            )

    def test_csa_share_happy_path(self):
        # 1915-1918: a -1 layer satisfies the CSA kinds guard.
        config = TransformerConfig(
            num_hidden_layers=2,
            experimental_attention_variant="dsv4_hybrid",
            csa_compress_ratios=[-1, 4],
            csa_share_docmask_meta=True,
            enable_hyper_connections=True,
        )
        self.assertTrue(config.csa_share_docmask_meta)

    def test_mqa_share_requires_latent_mqa(self):
        # 1934-1935: the default 'mha' builds dense MLA, the switch is a no-op.
        with self.assertRaises(ValueError):
            TransformerConfig(
                num_hidden_layers=2,
                experimental_attention_variant="dsv4_hybrid",
                csa_compress_ratios=[-2, -2],
                mqa_share_docmask_meta=True,
                enable_hyper_connections=True,
                hybrid_mla_attention="mha",
                hybrid_mla_q_lora_rank=128,
                hybrid_mla_kv_lora_rank=512,
                hybrid_mla_qk_nope_head_dim=192,
                hybrid_mla_qk_rope_head_dim=64,
                hybrid_mla_v_head_dim=64,
                hybrid_mla_num_attention_heads=1,
                hybrid_mla_num_key_value_heads=1,
            )

    def test_mqa_share_happy_path(self):
        # 1929-1933: -2 layer under mqa_dsa satisfies the latent-MQA guard.
        config = TransformerConfig(
            num_hidden_layers=2,
            experimental_attention_variant="dsv4_hybrid",
            csa_compress_ratios=[-2, 4],
            mqa_share_docmask_meta=True,
            enable_hyper_connections=True,
            hybrid_mla_attention="mqa_dsa",
            hybrid_mla_q_lora_rank=128,
            hybrid_mla_kv_lora_rank=512,
            hybrid_mla_qk_nope_head_dim=192,
            hybrid_mla_qk_rope_head_dim=64,
            hybrid_mla_v_head_dim=64,
            hybrid_mla_num_attention_heads=1,
            hybrid_mla_num_key_value_heads=1,
            dsa_index_n_heads=64,
            dsa_index_head_dim=128,
            dsa_index_topk=128,
        )
        self.assertTrue(config.mqa_share_docmask_meta)


def _layer(is_mtp=False, layer_number=1, share=True):
    cfg = TransformerConfig(
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=1,
    )
    cfg.csa_share_docmask_meta = share
    cfg.mqa_share_docmask_meta = False
    spec = TransformerLayerSublayersSpec(
        self_attention_hyper_connection=IdentityFuncOp,
        mlp_hyper_connection=IdentityFuncOp,
    )
    return HyperConnectionTransformerLayer(
        config=cfg,
        sublayers_spec=spec,
        layer_number=layer_number,
        pg_collection=_FakePGCollection(),
        is_mtp_layer=is_mtp,
    )


class TestDocMaskLayerWiring(unittest.TestCase):
    def setUp(self):
        doc_mask_meta_registry.begin_step(_ACC)

    def test_switch_on_registers_consumer_and_advances(self):
        layer = _layer(share=True)
        # ctor registered (1, False); advance 0 -> 1, then audit accepts
        self.assertEqual(layer._docmask_meta_key, (1, False))
        self.assertIn((1, False), doc_mask_meta_registry._cnt)
        kw1 = layer._docmask_meta_kwargs()
        kw2 = layer._docmask_meta_kwargs()
        self.assertEqual(kw1, {"docmask_mb_idx": 0})
        self.assertEqual(kw2, {"docmask_mb_idx": 1})
        doc_mask_meta_registry.check()

    def test_switch_off_registers_nothing_and_returns_empty(self):
        # a fresh layer_number avoids the singleton keys registered by other
        # tests in the same process (begin_step resets counts, not key sets)
        layer = _layer(share=False, layer_number=99)
        self.assertEqual(layer._docmask_meta_key, (99, False))
        self.assertNotIn((99, False), doc_mask_meta_registry._cnt)
        self.assertEqual(layer._docmask_meta_kwargs(), {})

    def test_mtp_layer_uses_its_own_consumer_key(self):
        layer = _layer(is_mtp=True, layer_number=4)
        self.assertEqual(layer._docmask_meta_key, (4, True))
        self.assertIn((4, True), doc_mask_meta_registry._cnt)
        # separate counter from the main layer
        main = _layer(is_mtp=False, layer_number=1)
        self.assertEqual(main._docmask_meta_kwargs(), {"docmask_mb_idx": 0})
        self.assertEqual(layer._docmask_meta_kwargs(), {"docmask_mb_idx": 0})

    def test_advance_respects_accumulation_boundary(self):
        layer = _layer(share=True)
        doc_mask_meta_registry.begin_step(_ACC)
        got = [layer._docmask_meta_kwargs()["docmask_mb_idx"] for _ in range(5)]
        self.assertEqual(got, [0, 1, 0, 1, 0])


def _module(is_mtp=False):
    cfg = _create_mqa_config("mqa_dsa", loss_coeff=0.0)
    # ``mqa_dsa`` fixture: indexer present, sparse loss on by default.
    return _build_module(
        cfg, bf16=True, is_mtp=is_mtp, pg_collection=_FakePGCollection()
    )


def _preload():
    doc_mask_meta_registry.begin_step(2)
    for mb in range(2):
        doc_mask_meta_registry.preload_mqa(
            mb, 1, _SEQ, _row_end([_SEQ], _SEQ), ("main",), WINDOW
        )


def _forward(module, mb_idx, switch):
    module.config.mqa_share_docmask_meta = switch
    query, key, w_v = _make_inputs(_SEQ, seed=3)
    q, k, wv = [t.clone().detach() for t in (query, key, w_v)]
    for t in (q, k, wv):
        t.stop_gradient = False
    return module(
        q,
        k,
        None,
        None,
        _row_end([_SEQ], _SEQ),
        v_b_proj_weight=wv,
        docmask_mb_idx=mb_idx,
    )


@unittest.mock.patch.object(
    MQALatentAttention, "_forward_sparse", return_value=None
)
class TestMQASlotLookup(unittest.TestCase):
    """forward's slot lookup with the phase forwards stubbed."""

    def test_preloaded_slot_is_handed_to_sparse(self, sparse):
        module = _module()
        _preload()
        _forward(module, mb_idx=0, switch=True)
        self.assertTrue(sparse.called)
        meta = sparse.call_args.kwargs["meta"]
        self.assertIsNotNone(meta)
        self.assertEqual(meta.seqlen, _SEQ)

    def test_switch_off_reads_nothing(self, sparse):
        module = _module()
        _preload()
        _forward(module, mb_idx=0, switch=False)
        self.assertTrue(sparse.called)
        self.assertIsNone(sparse.call_args.kwargs["meta"])

    def test_negative_idx_skips_lookup_even_with_switch_on(self, sparse):
        module = _module()
        _preload()
        _forward(module, mb_idx=-1, switch=True)
        self.assertTrue(sparse.called)
        self.assertIsNone(sparse.call_args.kwargs["meta"])

    def test_mtp_group_miss_falls_back_to_private_build(self, sparse):
        module = _module(is_mtp=True)
        _preload()  # only ("main", 0)/(..., 1) slots exist
        _forward(module, mb_idx=0, switch=True)
        self.assertTrue(sparse.called)
        self.assertIsNone(sparse.call_args.kwargs["meta"])


class TestEmbeddingInitBranch(unittest.TestCase):
    """``transformer_config.py:1762-1784``: the embedding init method branch.

    ``use_truncated_normal_init`` / ``magic_init`` are off by default, so every
    plain construction walks the ``elif self.embedding_init_method is None:``
    arm. The inner ``if`` splits on whether the requested init std differs from
    the model-wide one.
    """

    def test_default_uses_embedding_std_when_init_unset(self):
        # 1771-1778: ``init_method`` unset -> build ``init_method_normal`` from
        # ``embedding_init_method_std`` alone (here a non-default 0.04).
        config = TransformerConfig(
            num_hidden_layers=2, embedding_init_method_std=0.04
        )
        # ``init_method_normal`` returns a ``functools.partial`` (no __eq__);
        # inspect the bound keyword args instead of comparing objects.
        self.assertEqual(
            config.embedding_init_method.keywords, {"mean": 0.0, "std": 0.04}
        )

    def test_reuses_init_method_when_stds_match(self):
        # 1779-1784: explicit init + same std keeps the requested method, so an
        # embedding carries exactly the model-wide initialization.
        init = init_method_normal(0.02)
        config = TransformerConfig(
            num_hidden_layers=2,
            init_method=init,
            embedding_init_method_std=0.02,
        )
        self.assertIs(config.embedding_init_method, init)

    def test_hybrid_mla_ratio_counts_loop_runs_on_valid_config(self):
        # 1806-1811: a fully valid dsv4_hybrid + latent-MQA construction must
        # walk the ratio_counts loop (int keys via ``__index__``) and satisfy
        # the "-2 presence" guard without raising.
        config = TransformerConfig(
            num_hidden_layers=2,
            experimental_attention_variant="dsv4_hybrid",
            hybrid_mla_attention="mqa_dsa",
            csa_compress_ratios=[-2, 4],
            hybrid_mla_q_lora_rank=128,
            hybrid_mla_kv_lora_rank=512,
            hybrid_mla_qk_nope_head_dim=192,
            hybrid_mla_qk_rope_head_dim=64,
            hybrid_mla_v_head_dim=64,
            hybrid_mla_num_attention_heads=8,
            hybrid_mla_num_key_value_heads=8,
            dsa_index_n_heads=64,
            dsa_index_head_dim=128,
            dsa_index_topk=128,
        )
        self.assertEqual(config.csa_compress_ratios, [-2, 4])
        self.assertEqual(config.hybrid_mla_attention, "mqa_dsa")


class _StubCoreAttention:
    """``core_attention`` replacement for the white-box MLA probe.

    ``config`` deliberately carries no ``forward_meta`` (plain MLA, not the
    flash-decoding variant), so the forward takes the ``q_absorbed = None``
    branch; ``__call__`` records the kwarg pass-through and returns a tensor.
    """

    def __init__(self):
        self.config = SimpleNamespace()
        self.calls = 0
        self.last_kwargs = None

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return paddle.zeros([1, 1, 1])


class _InferenceKwargsMLA(MultiLatentAttention):
    """White-box probe for the base ``MultiLatentAttention.forward`` template.

    The production MLA classes (``MLASelfAttention`` / ``MQASelfAttention``)
    carry their own dsv4 ``forward``; this probe inherits the base template so
    the shared inference-kwargs extraction
    (``multi_latent_attention.py:950-957``) executes on CPU. The parent
    constructor (parameters + kernel handles) is replaced by the minimal
    attribute set the template reads.
    """

    def __init__(self, config):
        # ``paddle.nn.Layer.__init__`` only (``_forward_pre_hooks`` etc.);
        # ``Attention``'s chain builds parameters and kernel handles.
        paddle.nn.Layer.__init__(self)
        self.config = config
        self.layer_number = 1
        self.attn_mask_type = AttnMaskType.causal
        self.mqa_latent = False
        self.mqa_latent_split_kv_b = False
        self.recompute_core_attention = False
        self.recompute_qkv_up_porj_and_rope = False
        self.use_vha_postmix = False
        self.gated_attention = False
        self.use_rr_flash_attention = False
        self.o_proj = lambda out: (out, None)

    def get_query_key_value_tensors(self, *args, **kwargs):
        # ``Attention`` declares this abstract; the tests patch it with the
        # mocked qkv tuple, so this body is never reached.
        raise NotImplementedError(
            "patched by TestMultiLatentForwardInferenceKwargs"
        )


class TestMultiLatentForwardInferenceKwargs(unittest.TestCase):
    """The base MLA template forward's kwargs extraction + ``is_decode``.

    Runs the probe's full forward on CPU: ``get_query_key_value_tensors`` is
    mocked so nothing touches a projector, ``core_attention`` is the stub above,
    and the checks assert the kwargs really made it to the core attention call
    (``multi_latent_attention.py:952-957``).
    """

    def setUp(self):
        self.module = _InferenceKwargsMLA(
            TransformerConfig(
                num_hidden_layers=2, hidden_size=8, num_attention_heads=1
            )
        )
        self.module.core_attention = _StubCoreAttention()
        self.qkv = tuple(paddle.zeros([1, 8, 1, 4]) for _ in range(6))

    def patch_qkv(self):
        return unittest.mock.patch.object(
            self.module,
            "get_query_key_value_tensors",
            return_value=self.qkv,
        )

    def test_prefill_runs_extraction_with_use_cache_unset(self):
        # Training prefill: no cache kwargs at all -> is_decode short-circuits
        # False on the first ``_is_incremental_decode`` guard.
        with self.patch_qkv():
            output, bias = self.module(
                paddle.zeros([1, 8, 4]),
                None,
                key_value_states=None,
                position_ids=None,
                shared_kv=None,
            )
        self.assertEqual(self.module.core_attention.calls, 1)
        self.assertEqual(output.shape, [1, 1, 1])
        self.assertIsNone(bias)
        self.assertIsNone(
            self.module.core_attention.last_kwargs["past_key_values"]
        )
        self.assertIsNone(self.module.core_attention.last_kwargs["layer_idx"])
        self.assertFalse(self.module.core_attention.last_kwargs["use_cache"])

    def test_decode_cache_kwargs_thread_through(self):
        # Decode: past_key_values + layer_idx + use_cache=True -> is_decode
        # consults has_layer_cache(layer_idx) and returns True.
        cache = SimpleNamespace(has_layer_cache=lambda idx: True)
        with self.patch_qkv():
            self.module(
                paddle.zeros([1, 8, 4]),
                None,
                past_key_values=cache,
                layer_idx=3,
                use_cache=True,
            )
        self.assertEqual(self.module.core_attention.calls, 1)
        self.assertIs(
            self.module.core_attention.last_kwargs["past_key_values"], cache
        )
        self.assertEqual(self.module.core_attention.last_kwargs["layer_idx"], 3)
        self.assertTrue(self.module.core_attention.last_kwargs["use_cache"])

    def test_docmask_mb_idx_reaches_core_attn_extra(self):
        # multi_latent_attention.py:973: on a latent-MQA module the slot kwarg
        # forwarded by the layer must land in core_attn_extra, hence in the
        # core attention call's kwargs. Sets the mqa_latent branch with a stub
        # kv_b_proj so the whole wv_b extraction runs on CPU.
        module = self.module
        module.mqa_latent = True
        module.kv_lora_rank = 4
        module.qk_nope_head_dim = 4
        module.v_head_dim = 4
        module.num_attention_heads_per_partition = 1
        module.kv_b_proj = SimpleNamespace(weight=paddle.zeros([4, 8]))
        with self.patch_qkv():
            module(paddle.zeros([1, 8, 4]), None, docmask_mb_idx=0)
        self.assertEqual(module.core_attention.calls, 1)
        self.assertEqual(module.core_attention.last_kwargs["docmask_mb_idx"], 0)


class _DSv4SharedSlotProbe(DSv4HybridAttention):
    """White-box probe for ``DSv4HybridAttention.forward``'s shared-slot
    lookup (``dsv4_hybrid_attention.py:933-953``).

    Replaces the parent constructor with the small attribute set that forward
    reads before the lookup; the packed-batch helper and the down-stream
    ``_full_attn_forward`` are patched per test.
    """

    def __init__(self, config):
        paddle.nn.Layer.__init__(self)
        self.config = config
        self.layer_number = 1
        self.recompute_full_attn = False
        self.csa_mask_group = ("main",)
        self.core_attention = SimpleNamespace(compress_ratio=4)
        self.o_proj = lambda out: (out, None)


class TestDSv4SharedDocSlot(unittest.TestCase):
    """The CSA slot lookup inside ``DSv4HybridAttention.forward`` on CPU."""

    def setUp(self):
        doc_mask_meta_registry.begin_step(_ACC)
        config = TransformerConfig(
            num_hidden_layers=2, hidden_size=8, num_attention_heads=1
        )
        config.csa_share_docmask_meta = True
        config.csa_dense_mode = False
        self.module = _DSv4SharedSlotProbe(config)

    def test_forward_consumes_prebuilt_csa_slot(self):
        doc_mask_meta_registry.preload(
            0,
            4,
            1,
            _SEQ,
            _row_end([_SEQ], _SEQ),
            dense_mode=False,
            mask_group=("main",),
            window_size=WINDOW,
        )
        with (
            unittest.mock.patch.object(
                self.module,
                "_full_attn_forward",
                return_value=paddle.zeros([1, _SEQ, 8]),
            ) as full,
            unittest.mock.patch(
                "paddlefleet.transformer.dsv4_hybrid_attention."
                "_pack_dsv4_logical_batch",
                side_effect=lambda hidden, row_end, **kw: (
                    hidden,
                    row_end,
                    int(hidden.shape[0]),
                    int(hidden.shape[1]),
                ),
            ),
        ):
            out, bias = self.module(
                paddle.zeros([1, _SEQ, 8]),
                attention_mask=None,
                attn_mask_startend_row_indices=_row_end([_SEQ], _SEQ),
                docmask_mb_idx=0,
            )
        self.assertEqual(full.call_count, 1)
        self.assertIsNone(bias)
        # the registry-provided slot metadata reaches the full-attn segment
        self.assertEqual(full.call_args[0][3].ratio, 4)
        self.assertEqual(out.shape, [1, _SEQ, 8])


def _meta_index_inputs(seqlen=4):
    """Shapes only: nothing below is numerically exercised -- every kernel
    leaf is stubbed, so zeros keep the test free of dtype / rotary coupling."""
    query = paddle.zeros([1, seqlen, 1, 64])
    kv = paddle.zeros([1, seqlen, 1, 64])
    x = paddle.zeros([1, seqlen, 8])
    qr = paddle.zeros([1, seqlen, 8])
    w_v = paddle.zeros([8, 1, 4])
    return query, kv, x, qr, w_v


def _doc_fields(row_end, seqlen):
    from paddlefleet.transformer.csa_attention import (
        _derive_csa_doc_boundaries,
    )

    return _derive_csa_doc_boundaries(row_end, seqlen)


def _stub_indexer_loss_leaves(module):
    """Common kernel-loss leaves for the two meta-lookup phase forwards.

    ``_needs_indexer_loss`` is forced on so the loss branch runs; projections /
    target / loss-mask are replaced by shape-only zeros. The two remaining
    kernel patches (the indexer topk kernel and the tilelang autoscaler) stay
    at the call site so the with-block reads linearly.
    """
    module._needs_indexer_loss = lambda: True
    module._indexer_projections = lambda *a, **k: (
        paddle.zeros([1, 4, 8]),
        paddle.zeros([1, 4, 8]),
        paddle.zeros([1, 4, 8]),
    )
    module._attn_target = lambda *a, **k: paddle.zeros([1, 4, 128])
    module._indexer_loss_mask = lambda *a, **k: (None, None)


_AUTOSCALER_PATCH = unittest.mock.patch(
    "paddlefleet.transformer.mqa_latent_attention."
    "TileLangCSAIndexerLossAutoScaler",
    autospec=True,
)


class TestMQAWarmupMetaLookup(unittest.TestCase):
    """``_forward_warmup``'s meta on/off indexer-range split
    (``mqa_latent_attention.py:889-894``), CPU with kernel leaves stubbed."""

    def _run(self, meta):
        module = _module()
        # The attention half of the warmup is dense FA4: stub the whole
        # ``_forward_full_causal`` (whose kernel path is GPU-gated), keep the
        # indexer-range split and the KL machinery real.
        module._forward_full_causal = lambda *a, **k: paddle.zeros([1, 4, 1])
        module._check_tilelang_indexer_support = lambda *a, **k: None
        _stub_indexer_loss_leaves(module)
        query, kv, x, qr, wv = _meta_index_inputs()
        row_end = _row_end([4], 4)
        doc_start, doc_len, is_valid, _, doc_starts = _doc_fields(row_end, 4)
        with (
            unittest.mock.patch(
                "paddlefleet.tilelang_ops.csa_indexer_topk_fwd",
                return_value=(
                    paddle.zeros([1, 4, 128], dtype="int32"),
                    paddle.zeros([1, 4, 128]),
                ),
            ),
            _AUTOSCALER_PATCH as scaler,
        ):
            scaler.apply.side_effect = lambda *a, **k: a[0]
            out = module._forward_warmup(
                query,
                kv,
                x,
                qr,
                wv,
                doc_start,
                doc_len,
                is_valid,
                doc_starts,
                32,  # kv_lora_rank
                None,  # input_ids
                0,  # position_offset
                4,  # s_local
                4,  # s_global
                row_end,
                meta,
            )
        self.assertEqual(out.shape, [1, 4, 1])

    def test_warmup_meta_branch(self):
        self._run(MQADocMeta.build(_row_end([4], 4), 1, 4))

    def test_warmup_private_branch(self):
        self._run(None)


class TestMQASparseMetaLookup(unittest.TestCase):
    """``_forward_sparse``'s meta on/off split (``mqa_latent_attention.py:1327
    --1343``), CPU with kernel leaves stubbed."""

    def _run(self, meta, cp=False):
        module = _module()
        # index_topk=128 is not in _LSE_INDEXER_TOPKS, so the else branch
        # assigns ``core_out = self._sparse_attn(...)`` without unpacking;
        # [b, s, h*v] matches what the following _deabsorb reshapes from.
        module._sparse_attn = lambda *a, **k: paddle.zeros([1, 4, 8])
        module.cp_enabled = cp
        _stub_indexer_loss_leaves(module)
        query, kv, x, qr, wv = _meta_index_inputs()
        row_end = _row_end([4], 4)
        doc_start, doc_len, is_valid, doc_lens, _ = _doc_fields(row_end, 4)
        with (
            unittest.mock.patch(
                "paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn."
                "cudnn_indexer_topk_fwd",
                return_value=(
                    paddle.zeros([1, 4, 128], dtype="int32"),
                    None,
                    paddle.zeros([1, 4, 128]),
                ),
            ),
            _AUTOSCALER_PATCH as scaler,
        ):
            scaler.apply.side_effect = lambda *a, **k: a[0]
            out = module._forward_sparse(
                query,
                kv,
                x,
                qr,
                wv,
                doc_start,
                doc_len,
                is_valid,
                doc_lens,  # global lengths -> the cudnn fast-path doc_lens
                512,  # kv_lora_rank
                input_ids=None,
                position_offset=0,
                meta=meta,
            )
        self.assertEqual(out.shape, [1, 4, 4])

    def test_sparse_meta_branch(self):
        self._run(MQADocMeta.build(_row_end([4], 4), 1, 4))

    def test_sparse_private_branch(self):
        self._run(None)

    def test_sparse_private_cp_row_slice(self):
        # 1337-1340: with cp_enabled the privately-built window table is
        # row-sliced to the rank's local sequence before the kernel call.
        self._run(None, cp=True)


class _ProbeMHC(paddle.nn.Layer):
    """mHC aggregation stand-in: ``(aggregated, h_res, h_post) = (x, x, x)``."""

    def forward(self, x):
        return x, x, x


class _ProbeCrossAttention(paddle.nn.Layer):
    """Cross attention stand-in returning ``(hidden, bias=None)``."""

    def forward(self, x, **kwargs):
        return x, None


class _ProbeBias(paddle.nn.Layer):
    """Cross-attn bias-dropout stand-in: ``fn(training, fuse)`` must return a
    callable; the call site then applies it as ``fn(out, residual, p)``."""

    def forward(self, *args, **kwargs):
        return self._apply

    def _apply(self, out, residual, p):
        return out[0] if isinstance(out, tuple) else out


class TestLayerDocmaskSlotKwarg(unittest.TestCase):
    """``transformer_layer.py:1917-1920``: the mHC ``_forward_attention``
    forwards the shared docmask slot kwarg to an MLA-family self-attention
    module, backed by the white-box MLA probe."""

    def test_forward_attention_threads_slot_kwarg(self):
        layer = _layer(share=False, layer_number=5)
        probe = _InferenceKwargsMLA(layer.config)
        probe.core_attention = _StubCoreAttention()
        layer.self_attn = probe
        # Registered sublayers must stay paddle layer types; the real
        # layernorms run fine on CPU, the hyper-connection and cross attention
        # are replaced, and the fusion helpers (plain methods) are overridden.
        layer.self_attention_hyper_connection = _ProbeMHC()
        layer.cross_attention = _ProbeCrossAttention()
        layer.cross_attn_bda = _ProbeBias()
        layer._fused_h_res_h_post_bda = lambda *a, **k: (
            a[4][0] if isinstance(a[4], tuple) else a[4],
            None,
        )
        layer._cast_and_discard_fused_bda = lambda out, *a, **k: out
        qkv = tuple(paddle.zeros([1, 8, 1, 4]) for _ in range(6))
        with (
            unittest.mock.patch.object(
                probe, "get_query_key_value_tensors", return_value=qkv
            ),
            unittest.mock.patch.object(
                probe, "forward", wraps=probe.forward
            ) as fwd,
        ):
            layer._forward_attention(
                paddle.zeros([1, 8, 8]),
                attention_mask=None,
                docmask_mb_idx=0,
            )
        self.assertEqual(probe.core_attention.calls, 1)
        # the slot kwarg made it into the self-attention call (line 1920)
        self.assertEqual(fwd.call_args.kwargs["docmask_mb_idx"], 0)
