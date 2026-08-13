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

``hybrid_mla_attention`` set to ``"mqa_dsa"`` or ``"mqa_full_causal"`` turns the
hybrid MLA (``csa_compress_ratios == -2``) layers of a ``dsv4_hybrid`` model into
:class:`MQALatentAttention` (latent MQA). The module picks its path from the
sublayers spec, not from any config string:

* ``MQALatentAttentionSublayersSpec(indexer=None)`` -- per-document full-causal
  attention on the latent, mathematically equal to MHA. This is what production
  builds for ``"mqa_full_causal"``; ``gpt_layer_specs`` always attaches an
  indexer for ``"mqa_dsa"``. The absorption-equivalence tests here drive it by
  constructing the layer directly with ``indexer=None``.
* an indexer spec -- forced local window + Lightning-indexer top-k, i.e. DSA on
  the KV latent.

Coverage:
  1. Guards -- unsupported configurations fail loudly (no GPU needed).
  2. Index construction over adversarial multi-document layouts: the forced
     128-window and the indexer candidate range are disjoint yet jointly equal
     the per-document causal set (no duplicate column, no lost window column).
  3. The indexer-less full-causal path equals a dense fp32 reference, because
     the activation-level absorption is exactly score preserving.
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
  7. The phase-2 (warmup) shape of ``"mqa_dsa"``, selected by
     ``dsa_indexer_use_sparse_loss=False``: attention consumes the full
     per-document causal table (bit-identical to ``"mqa_full_causal"``) while
     the indexer's top-k serves the wide KL loss only.
  8. Migration: the renamed config keys (``non_absorbed_mqa*``,
     ``csa_train_indexer_only``, ``csa_indexer_init_from_scratch``) ship without
     an alias, so a stale config must raise rather than be absorbed into a
     silent default.
  9. The fused indexer-loss target's plumbing with both kernels stubbed out:
     the ``_attn_target`` dispatch, ``_attn_target_cudnn``'s call contract and
     empty-slot handling, ``mqa_sparse_attn``'s ``lse_indexer`` side channel and
     the ``_forward_sparse`` branch that asks for it. The kernels themselves need
     SM100+ (see 5 and ``TestMQADSACudnnTarget``); everything around them is
     plain Python and stays checked on machines that lack them.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle

from paddlefleet.transformer.csa_attention import (
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
)
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.mqa_latent_attention import (
    _LSE_INDEXER_TOPKS,
    MQALatentAttention,
    _HashableTensor,
)
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
    _fa4_module_hooks,
    _full_causal_indices,
    _make_inputs,
    _rel,
    _row_end,
)

setUpModule, tearDownModule = _fa4_module_hooks()

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


def _full_causal_table(layout, seqlen):
    """The per-document full-causal ``[1, s, s]`` table.

    A pure integer function of the document bounds, shared with the CP suites
    (``hybrid_mla_utils._full_causal_indices``). Production never materialises
    it -- the dense FA4 backend gets the same column set as an ``O(s)`` row
    bound -- so this is an independent derivation of what that mask must be.
    """
    row_end = _row_end(layout, seqlen)
    doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, seqlen)
    table = _full_causal_indices(1, seqlen, doc_start, is_valid)
    return table.numpy()


def _fp32(tensor):
    """bf16 -> fp32 numpy; the widening is exact, so bit equality survives."""
    return tensor.cast("float32").numpy()


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

    def test_v_b_proj_weight_layout_mismatch_rejected(self):
        # Both layouts reshape fine, so a ``[h, v, l]`` weight handed to the
        # einsum path (or the reverse) would silently mis-compute. The
        # contraction dim is checked against the config rank instead.
        query, key, _ = self._args()
        w_v = paddle.zeros([H, V_HEAD_DIM, DV], dtype="bfloat16")
        with self.assertRaises(ValueError):
            self.module(query, key, None, None, v_b_proj_weight=w_v)

    def test_v_b_proj_weight_rank_mismatch_rejected(self):
        # A folded 2-D parameter that was never reshaped back must be named as
        # such, not fail later on an unpacking whose message hides the cause.
        query, key, _ = self._args()
        w_v = paddle.zeros([H * V_HEAD_DIM, DV], dtype="bfloat16")
        with self.assertRaisesRegex(ValueError, "must be 3-D"):
            self.module(query, key, None, None, v_b_proj_weight=w_v)

    def test_kv_lora_rank_comes_from_the_hybrid_field(self):
        # The rank is not derivable from ``v_b_proj_weight.shape[0]`` once the
        # grouped-matmul layout is in play, so the layer reads it from the
        # config: the hybrid field when set, the model-wide one otherwise.
        self.assertEqual(self.module.kv_lora_rank, DV)
        config = _create_mqa_config("mqa")
        config.hybrid_mla_kv_lora_rank = None
        config.kv_lora_rank = DV
        self.assertEqual(_build_module(config).kv_lora_rank, DV)

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
    """The hybrid MLA config surface after the ``hybrid_mla_attention`` refactor.

    The old 3-state ``hybrid_mla_attn_mode`` and the ``hybrid_mla_attn_sink``
    switch (with its mutual-exclusion ValueError against the model-wide sinks)
    are gone. There is now a single enum ``hybrid_mla_attention`` (``"mha"`` /
    ``"mqa_dsa"`` / ``"mqa_full_causal"``) and a single sink switch
    (``add_full_attention_sink_bias`` / ``softmax_type``), so the two can no
    longer conflict. Under ``"mqa_dsa"`` the -2 layers run a cuDNN DSA indexer,
    so the config validates the model-wide ``dsa_index_*`` fields
    (index_n_heads / index_head_dim / index_topk).
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
    def _mqa_dsa_kwargs(cls, **overrides):
        # ``hybrid_mla_attention="mqa_dsa"`` triggers the DSA-indexer
        # validation, so a valid baseline must carry the model-wide index dims.
        base = {
            "hybrid_mla_attention": "mqa_dsa",
            "dsa_index_n_heads": INDEX_HEADS,
            "dsa_index_head_dim": INDEX_HEAD_DIM,
            "dsa_index_topk": INDEX_TOPK,
        }
        base.update(overrides)
        return cls._kwargs(**base)

    def test_hybrid_mla_attention_defaults_to_mha(self):
        config = TransformerConfig(**self._kwargs())
        self.assertEqual(config.hybrid_mla_attention, "mha")

    def test_mqa_dsa_accepted_with_valid_index_dims(self):
        config = TransformerConfig(**self._mqa_dsa_kwargs())
        self.assertEqual(config.hybrid_mla_attention, "mqa_dsa")
        self.assertEqual(config.dsa_index_head_dim, 128)

    def test_sink_coexists_with_latent_mqa(self):
        # The old mutual-exclusion ValueError is gone: one sink switch only, so
        # enabling a model-wide sink alongside latent MQA must be accepted.
        for sink in (
            {"add_full_attention_sink_bias": True},
            {"softmax_type": "learnable"},
        ):
            with self.subTest(sink=sink):
                config = TransformerConfig(**self._mqa_dsa_kwargs(**sink))
                self.assertEqual(config.hybrid_mla_attention, "mqa_dsa")

    def test_index_dims_are_validated(self):
        # Keyed on (field, value), not on the message: topk=100 (not a multiple
        # of 128) and topk=2176 (>2048) both mention "index_topk" but hit
        # different branches.
        #
        # ``index_topk`` is validated in the sparse phase only, because that is
        # the only phase that reads it -- the warmup KL spans the whole causal
        # set and runs no top-k -- so the topk cases must ask for that phase.
        for field, value, msg in (
            ("dsa_index_head_dim", 64, "index_head_dim"),
            ("dsa_index_topk", 100, "index_topk"),
            ("dsa_index_topk", 2048 + 128, "index_topk"),
            ("dsa_index_n_heads", None, "index_n_heads"),
        ):
            extra = (
                {"dsa_indexer_use_sparse_loss": True}
                if field == "dsa_index_topk"
                else {}
            )
            with (
                self.subTest(field=field, value=value),
                self.assertRaisesRegex(ValueError, msg),
            ):
                TransformerConfig(
                    **self._mqa_dsa_kwargs(**{field: value}, **extra)
                )

    def test_warmup_phase_needs_no_index_topk(self):
        """Phase 2 must not be forced to carry a top-k budget.

        ``_forward_warmup`` never selects a top-k, so a kernel-illegal (or
        simply absent, hence default) ``index_topk`` must not block startup --
        while the sparse phase still rejects it. The production phase-2
        ``model_config.json`` relies on this: it ships no ``index_topk`` at all.
        """
        for topk in (100, 2048 + 128):
            with self.subTest(dsa_index_topk=topk):
                config = TransformerConfig(
                    **self._mqa_dsa_kwargs(
                        dsa_index_topk=topk,
                        dsa_indexer_use_sparse_loss=False,
                    )
                )
                self.assertEqual(config.hybrid_mla_attention, "mqa_dsa")
                self.assertFalse(config.dsa_indexer_use_sparse_loss)
                # ...and the same value is still rejected in the sparse phase,
                # so this is a phase gate, not a dropped check.
                with self.assertRaisesRegex(ValueError, "index_topk"):
                    TransformerConfig(
                        **self._mqa_dsa_kwargs(
                            dsa_index_topk=topk,
                            dsa_indexer_use_sparse_loss=True,
                        )
                    )

    def test_illegal_hybrid_mla_attention_configs_are_rejected(self):
        """The enum makes the old ``(dense=True, mqa=False)`` state
        unrepresentable, so what is left to reject is (a) a value outside the
        enum and (b) a latent-MQA mode on a config that owns no MLA (``-2``)
        layer -- which used to be a silent no-op. Both must be config errors.
        """
        # (a) out-of-enum values, including near-misses and the old bool.
        for value in ("mqa", "MHA", "dense", "", None, True):
            with (
                self.subTest(hybrid_mla_attention=value),
                self.assertRaisesRegex(ValueError, "is invalid"),
            ):
                TransformerConfig(**self._kwargs(hybrid_mla_attention=value))
        # (b) a latent-MQA mode with no -2 layer to apply to. Ratio -1 is CSA
        # full-causal MQA, a different layer kind, so it must not satisfy the
        # check either; nor may a non-dsv4_hybrid variant.
        for mode in ("mqa_dsa", "mqa_full_causal"):
            for ratios, variant in (
                ([128, 128], "dsv4_hybrid"),
                ([-1, -1], "dsv4_hybrid"),
                (None, "dsv4_hybrid"),
                ([-2, -2], None),
            ):
                with (
                    self.subTest(mode=mode, ratios=ratios, variant=variant),
                    self.assertRaisesRegex(
                        ValueError, "only applies to MLA layers"
                    ),
                ):
                    TransformerConfig(
                        **self._kwargs(
                            hybrid_mla_attention=mode,
                            csa_compress_ratios=ratios,
                            experimental_attention_variant=variant,
                            dsa_index_n_heads=INDEX_HEADS,
                            dsa_index_head_dim=INDEX_HEAD_DIM,
                            dsa_index_topk=INDEX_TOPK,
                        )
                    )

    def test_split_kv_b_proj_only_means_anything_for_latent_mqa(self):
        # The switch splits latent MQA's kv_b_proj into standalone k_b_proj /
        # v_b_proj absorption parameters. On the dense MHA path there is nothing
        # to split, so silently accepting it would hide a mis-set config.
        with self.assertRaisesRegex(ValueError, "only means"):
            TransformerConfig(**self._kwargs(mqa_split_kv_b_proj=True))
        for mode in ("mqa_dsa", "mqa_full_causal"):
            with self.subTest(mode=mode):
                config = TransformerConfig(
                    **self._mqa_dsa_kwargs(
                        hybrid_mla_attention=mode,
                        mqa_split_kv_b_proj=True,
                    )
                )
                self.assertTrue(config.mqa_split_kv_b_proj)

    def test_split_kv_b_proj_rejects_hy_sparse_attention(self):
        # HySparse swaps the layer class for MQASelfAttention, whose forward and
        # decode paths still absorb against kv_b_proj.weight -- the parameter the
        # split removes. Accepting the combination would fail on a None
        # attribute deep in the forward.
        with self.assertRaisesRegex(ValueError, "enable_hy_sparse_attention"):
            TransformerConfig(
                **self._mqa_dsa_kwargs(
                    hybrid_mla_attention="mqa_dsa",
                    mqa_split_kv_b_proj=True,
                    enable_hy_sparse_attention=True,
                )
            )

    def test_mqa_full_causal_does_not_require_index_dims(self):
        # No indexer is built, so the index_* validation must be skipped -- these
        # kwargs deliberately omit dsa_index_* and would otherwise be rejected.
        config = TransformerConfig(
            **self._kwargs(hybrid_mla_attention="mqa_full_causal")
        )
        self.assertEqual(config.hybrid_mla_attention, "mqa_full_causal")

    def test_index_dims_unvalidated_when_hybrid_mla_attention_is_mha(self):
        """With the mode left at ``"mha"`` the -2 layers are dense per-head
        attention, so the indexer fields are unused and must not be validated.

        Asserting only that a *default* config builds is near-tautological: the
        defaults are ``None``, which is exactly what
        ``test_index_dims_are_validated`` shows the ``"mqa_dsa"`` path rejects,
        but nothing pins the other three rejections. So feed the exact values
        that test proves are rejected under ``"mqa_dsa"`` -- head_dim != 128,
        topk not a multiple of 128, topk > 2048 -- and assert each one builds
        and survives onto the config unchanged.
        """
        bad = {
            "dsa_index_n_heads": None,
            "dsa_index_head_dim": 64,
            "dsa_index_topk": 100,
        }
        for field, value in [*bad.items(), ("dsa_index_topk", 2048 + 128)]:
            with self.subTest(field=field, value=value):
                config = TransformerConfig(
                    **self._kwargs(hybrid_mla_attention="mha", **{field: value})
                )
                self.assertEqual(config.hybrid_mla_attention, "mha")
                self.assertEqual(getattr(config, field), value)
        # ... and all of them together, still no raise.
        config = TransformerConfig(
            **self._kwargs(hybrid_mla_attention="mha", **bad)
        )
        self.assertEqual(config.hybrid_mla_attention, "mha")

    def test_train_indexer_only_is_pinned_to_the_wide_indexer_loss(self):
        """The two phases are fixed pairs, not four independent modes.

        On the ``-2`` layers ``dsa_indexer_use_sparse_loss`` decides the
        attention candidate set as well as the KL width, so
        ``train_indexer_only=True`` (frozen backbone, warmup) only makes sense
        with the wide loss. The mixed pair is rejected; the other mix (trainable
        backbone + wide loss) is merely unusual and only warns, so it must still
        build.

        Constructed from scratch every time: ``__post_init__`` is not reentrant,
        so mutating an already-normalised config and re-validating would trip
        unrelated ``first_k_dense_replace`` / ``moe_layer_freq`` checks.
        """
        # ``train_indexer_only`` additionally demands a positive loss coeff, so
        # the legal pair carries one; the illegal pair is rejected before that
        # check is even reached (transformer_config.py:1540 vs :1708).
        with self.assertRaisesRegex(ValueError, "is not a valid phase"):
            TransformerConfig(
                **self._mqa_dsa_kwargs(
                    train_indexer_only=True,
                    dsa_indexer_use_sparse_loss=True,
                    dsa_indexer_loss_coeff=0.01,
                )
            )
        for indexer_only, sparse_loss in ((True, False), (False, True)):
            with self.subTest(
                train_indexer_only=indexer_only, sparse_loss=sparse_loss
            ):
                config = TransformerConfig(
                    **self._mqa_dsa_kwargs(
                        train_indexer_only=indexer_only,
                        dsa_indexer_use_sparse_loss=sparse_loss,
                        dsa_indexer_loss_coeff=0.01,
                    )
                )
                self.assertEqual(config.train_indexer_only, indexer_only)
                self.assertEqual(
                    config.dsa_indexer_use_sparse_loss, sparse_loss
                )

    def test_renamed_config_keys_are_rejected_not_absorbed(self):
        """A stale config key must fail loudly instead of turning into a no-op.

        The renames here ship without a compatibility alias (the repo's habit --
        see ``sonicmoe_quant_format``), so the only question is whether a config
        that still carries the old key is *told*. Two paths, two mechanisms:

        * direct construction -- the dataclass ``__init__`` already raises
          ``TypeError`` on an unknown kwarg, so nothing was needed;
        * :meth:`TransformerConfig.from_config` -- ``_process_attribute``'s
          fallback is a bare ``setattr``, so the stale key used to be absorbed
          as a dead attribute. The switch it was meant to flip stayed at its
          default and nothing complained: ``non_absorbed_mqa=True`` silently
          became ``hybrid_mla_attention="mha"``. That is the hole this pins.
        """
        legacy_to_new = {
            "non_absorbed_mqa": "hybrid_mla_attention",
            "non_absorbed_mqa_dense": "hybrid_mla_attention",
            "csa_train_indexer_only": "train_indexer_only",
            "csa_indexer_init_from_scratch": "indexer_init_from_scratch",
        }
        for legacy, replacement in legacy_to_new.items():
            # ``False`` must be rejected too: a stale key is a stale config even
            # when its value happens to agree with the new field's default, and
            # accepting it would leave the writer thinking the key still works.
            for value in (True, False):
                with self.subTest(legacy=legacy, value=value):
                    stale = SimpleNamespace(**self._kwargs(**{legacy: value}))
                    with self.assertRaises(ValueError) as raised:
                        TransformerConfig.from_config(stale)
                    message = str(raised.exception)
                    self.assertIn(f"{legacy} was renamed", message)
                    # The message has to name the replacement, otherwise the
                    # reader has to go read the diff to migrate.
                    self.assertIn(replacement, message)
                    with self.assertRaises(TypeError):
                        TransformerConfig(**self._kwargs(**{legacy: value}))

    def test_from_config_accepts_the_current_key_names(self):
        """Control for the test above: the rejection is keyed on the old names
        only, so the new ones must survive the same ``from_config`` path.
        """
        fresh = SimpleNamespace(
            **self._mqa_dsa_kwargs(
                train_indexer_only=True,
                dsa_indexer_use_sparse_loss=False,
                dsa_indexer_loss_coeff=0.01,
                indexer_init_from_scratch=True,
            )
        )
        config = TransformerConfig.from_config(fresh)
        self.assertEqual(config.hybrid_mla_attention, "mqa_dsa")
        self.assertTrue(config.train_indexer_only)
        self.assertTrue(config.indexer_init_from_scratch)


@_GPU
class TestMQAEquivalence(unittest.TestCase):
    """The indexer-less full-causal path is mathematically identical to MHA."""

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
        module = _build_module(_create_mqa_config("mqa"), bf16=True, sink=sink)
        # The dense FA4 backend takes the sink as ``learnable_sink`` and asserts
        # bf16 on it (``flash_mask/cute/interface.py:598``), which is what
        # ``build_softmax_offset``'s ``params_dtype`` gives in production. An
        # fp32 sink here would be testing a configuration the phase cannot run.
        self.assertEqual(module.softmax_offset.dtype, paddle.bfloat16)
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

    def test_use_sparse_loss_switches_both_attention_and_loss_width(self):
        """``dsa_indexer_use_sparse_loss`` picks the whole training phase.

        On these uncompressed ``-2`` layers the switch is one decision with two
        effects (``MQALatentAttention._phase``), not just the KL width it selects
        for the CSA layers of the same model
        (``_resolve_csa_indexer_loss_topk_effective``):

        * ``False`` -- phase 2 (warmup, ``_forward_warmup``). Attention consumes
          the **full per-document causal** table, because a freshly initialised
          indexer's ranking must not steer attention yet, and the KL is scored
          over that same full causal set -- so ``_attn_target``, the top-k KL
          target builder, is never called at all. (It used to be called with a
          *widened* top-k table; at the production ``index_topk=2048`` that
          widening degenerated back into the phase-3 table, which is why the
          phase now shares no loss code with phase 3.)
        * ``True`` -- phase 3 (``_forward_sparse``). Attention consumes
          ``window + index_topk`` and the KL is restricted to that same set, so
          ``_attn_target`` is called once per step at exactly ``index_topk``.

        Both column sets are asserted. The ``False`` branch has no column table
        of its own -- dense FA4 is its only backend and carries the mask as an
        ``O(s)`` row bound -- so ``RecordingMQA`` writes that bound out as the
        ``[b, s, s]`` set it denotes (``_row_end_column_table``). That decoding is
        a pure integer function of the document bounds with no floating-point
        scoring in it, so it is reproducible and *exactly* assertable, element for
        element, against ``_full_causal_table``.

        The ``True`` path stays statistical, which is the pre-existing measured
        fact this test still records: on a single full-length document neither
        the output bits nor the index table are reproducible across identical
        calls -- 2.4-2.8% of the table's slots move between two *identical*
        eval-mode calls, and ~1e-4 on the output. (Splitting the same 512 rows
        into two documents is exactly reproducible, which is why
        ``test_recompute_double_forward_is_consistent`` can assert equality --
        it uses ``[200, 312]``.)
        """
        seqlen = 512
        query, key, w_v, x, qr = self._inputs(seqlen, seed=5)
        loss_widths = []
        inner_target = self.module._attn_target

        def recording_target(query_, kv_, kl_columns, lse_indexer=None):
            # The KL's column set is the indexer's candidate set: the top-k in
            # the sparse phase, every causal column in warmup. The forced window
            # is never in it.
            loss_widths.append(int(kl_columns.shape[-1]))
            return inner_target(query_, kv_, kl_columns, lse_indexer)

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

        # The KL column set: exactly ``index_topk`` under ``True``; under
        # ``False`` the same builder is reached but over the whole causal span,
        # so the width is ``s``.
        self.assertEqual(loss_widths, [INDEX_TOPK, INDEX_TOPK, seqlen])
        # The KL never scores the forced window: its width is exactly the
        # indexer's candidate budget, not ``WINDOW + INDEX_TOPK``.
        self.assertNotIn(WINDOW + INDEX_TOPK, loss_widths)

        # The attention table: window + top-k under ``True``, the full causal
        # table (width ``s``) under ``False``.
        for table in (idx_a, idx_b):
            self.assertEqual(int(table.shape[-1]), WINDOW + INDEX_TOPK)
        self.assertEqual(list(idx_full.shape), [1, seqlen, seqlen])
        np.testing.assert_array_equal(
            idx_full, _full_causal_table([seqlen], seqlen)
        )

        # The measured identical-call drift of the ``True`` table, kept as the
        # reason its width -- not its contents -- is what gets asserted.
        drift = float((idx_a != idx_b).mean())
        self.assertLess(drift, 0.05)

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


@_GPU
class TestMQADSAWarmupPhase(unittest.TestCase):
    """Phase 2 of ``"mqa_dsa"``: ``dsa_indexer_use_sparse_loss=False``.

    The indexer is still being learned, so attention must not consume its
    ranking: it attends to the full per-document causal set (bit-identical to
    ``hybrid_mla_attention="mqa_full_causal"``) while the indexer's top-k feeds
    the wide KL loss only. ``TestMQADSA`` covers the phase-3 shape
    (``True``), where attention consumes ``window + index_topk``.

    Kept as its own class rather than folded into ``TestMQADSA``: the module
    fixture differs (the switch is off from construction, not flipped mid-test),
    and everything here is an exact assertion, because the full-causal table is
    integer-only.
    """

    SEQLEN = 256
    # Two documents, the second longer than the forced window, so the indexer's
    # candidate range is non-empty on the late rows yet still excludes them.
    LAYOUT = [40, 216]

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        self.module = self._build_warmup()
        self.row_end = _row_end(self.LAYOUT, self.SEQLEN)

    @staticmethod
    def _build_warmup():
        """A ``"mqa_dsa"`` module with the switch off from construction."""
        config = _create_mqa_config("mqa_dsa", loss_coeff=0.01)
        config.dsa_indexer_use_sparse_loss = False
        module = _build_module(config, bf16=True)
        assert module.indexer is not None
        assert module.indexer_use_sparse_loss is False
        return module

    def _inputs(self, seed=0):
        return _make_inputs(self.SEQLEN, seed=seed, with_hidden=True)

    def _call(self, module, tensors, w_v, training, differentiable=False):
        module.train() if training else module.eval()
        query, key, x, qr = tensors
        if differentiable:
            for tensor in tensors:
                tensor.stop_gradient = False
        return module(
            query,
            key,
            None,
            None,
            self.row_end,
            v_b_proj_weight=w_v,
            x=x,
            qr=qr,
        )

    def test_attention_output_equals_the_indexer_less_full_causal_path(self):
        """The core invariant: the indexer's *existence* must not move a bit.

        Both paths reach the same ``_forward_full_causal`` -- phase 1 directly
        (``:500``), the warmup through it (``:629``) -- and so the same dense FA4
        kernel with the same caller row bound, so the outputs must be
        bit-identical, not merely close. Nothing needs weight copying: on this
        path attention consumes no module parameter at all -- the
        query/key/``v_b_proj_weight`` are inputs and ``softmax_offset`` is
        ``None`` in both -- so the only thing that could differ is the mask.
        Asserted in both modes, so the warmup's ``:636`` eval early exit and its
        full indexer-loss branch are each covered.
        """
        query, key, w_v, x, qr = self._inputs()
        reference = _build_module(_create_mqa_config("mqa"), bf16=True)
        self.assertIsNone(reference.indexer)
        self.assertIsNone(reference.softmax_offset)
        self.assertIsNone(self.module.softmax_offset)
        self.assertEqual(self.module.softmax_scale, reference.softmax_scale)
        reference.eval()
        out_ref = _fp32(
            self._call(reference, (query, key, x, qr), w_v, training=False)
        )
        for training in (False, True):
            with self.subTest(training=training):
                DSAIndexerLossLoggingHelper.tracker.clear()
                tensors = [t.clone() for t in (query, key, x, qr)]
                out = self._call(
                    self.module,
                    tensors,
                    w_v,
                    training=training,
                    differentiable=training,
                )
                np.testing.assert_array_equal(_fp32(out), out_ref)

    def test_token_indices_are_the_full_causal_table(self):
        """The captured table is ``[b, s, s]`` and element-wise equal to the
        builder's own output, over several document layouts."""
        query, key, w_v, x, qr = self._inputs()
        for layout in ([self.SEQLEN], self.LAYOUT, [3, WINDOW, WINDOW + 1, 1]):
            with self.subTest(layout=layout):
                self.row_end = _row_end(layout, self.SEQLEN)
                _CAPTURED.clear()
                self._call(
                    self.module, (query, key, x, qr), w_v, training=False
                )
                table = _CAPTURED[-1]
                self.assertEqual(
                    list(table.shape), [1, self.SEQLEN, self.SEQLEN]
                )
                np.testing.assert_array_equal(
                    table, _full_causal_table(layout, self.SEQLEN)
                )
                # ... and the table is sound in its own right: the whole causal
                # set, no duplicate, nothing cross-document.
                _check_index_invariants(
                    self, table, self.row_end, self.SEQLEN, expect_full=True
                )

    def test_warmup_undoes_the_indexer_weight_prebake_for_tilelang(self):
        """The tilelang indexer re-applies ``head_dim**-0.5``, so the pre-bake
        must be undone -- exactly as the cuDNN pair needs in phase 3.

        This is a regression test for a bug the test suite was blind to: warmup
        first shipped passing ``weights`` through unscaled, on the (wrong)
        reasoning that the ``head_dim**0.5`` fixup was cuDNN-specific. Nothing
        crashed and every existing assertion still passed -- the indexer was just
        trained against a distribution flattened by ``1/sqrt(128)``. The precision
        audit caught it by comparing against a plain-paddle reference:
        un-baked weights match to ``max|d|=3.0e-8 / cosine 1-1.5e-13``, unscaled
        ones are off by ``max|d|=7.5e-1 / cosine 0.62``.

        So the discriminator has to be numeric. ``probs`` from the kernel is
        compared against the reference expression evaluated with ``weights`` **as
        ``forward_before_topk`` returns them**, which is the intended scale.
        """
        import paddle.nn.functional as F

        import paddlefleet.tilelang_ops as tl_mod

        seen = {}
        inner_tl = tl_mod.csa_indexer_topk_fwd
        inner_proj = self.module._indexer_projections

        def recording_proj(*args, **kwargs):
            q, k, w = inner_proj(*args, **kwargs)
            seen["w_as_returned"] = w.detach().cast("float32").numpy().copy()
            seen["q"] = q.detach().cast("float32").numpy().copy()
            seen["k"] = k.detach().cast("float32").numpy().copy()
            return q, k, w

        def recording_tl(*args, **kwargs):
            seen["w_passed"] = args[2].detach().cast("float32").numpy().copy()
            columns, probs = inner_tl(*args, **kwargs)
            seen["columns"] = columns.numpy().copy()
            seen["probs"] = probs.cast("float32").numpy().copy()
            return columns, probs

        query, key, w_v, x, qr = self._inputs()
        tensors = [t.clone() for t in (query, key, x, qr)]
        self.module._indexer_projections = recording_proj
        tl_mod.csa_indexer_topk_fwd = recording_tl
        try:
            self._call(
                self.module, tensors, w_v, training=True, differentiable=True
            )
        finally:
            self.module._indexer_projections = inner_proj
            tl_mod.csa_indexer_topk_fwd = inner_tl

        # The pre-bake is undone exactly once. ``weights`` is bf16, so the
        # product carries bf16 rounding (~2.6e-3 relative measured); the factor
        # being separated here is sqrt(128) ~ 11.3 against 1.0, so a loose
        # tolerance still discriminates it by three orders of magnitude.
        root_d = float(self.module.indexer.head_dim) ** 0.5
        np.testing.assert_allclose(
            seen["w_passed"],
            seen["w_as_returned"] * root_d,
            rtol=5e-3,
            atol=1e-6,
        )

        # ...and that is the scale which reproduces the reference distribution.
        q = paddle.to_tensor(seen["q"])
        k = paddle.to_tensor(seen["k"])
        w = paddle.to_tensor(seen["w_as_returned"])
        scores = paddle.einsum("bshd,btd->bsht", q, k)
        logits = (F.relu(scores) * w.unsqueeze(-1)).sum(axis=2)
        rows = paddle.arange(self.SEQLEN, dtype="int64").unsqueeze(-1)
        cols = paddle.arange(self.SEQLEN, dtype="int64").unsqueeze(0)
        doc_start = []
        start = 0
        for length in self.LAYOUT:
            doc_start += [start] * length
            start += length
        doc_start = paddle.to_tensor(doc_start, dtype="int64")
        keep = (cols <= rows) & (cols >= doc_start.unsqueeze(-1))
        logits = logits + paddle.where(
            keep.unsqueeze(0),
            paddle.zeros([1, self.SEQLEN, self.SEQLEN], dtype="float32"),
            paddle.full([1, self.SEQLEN, self.SEQLEN], -1e30, dtype="float32"),
        )
        ref = F.softmax(logits, axis=-1).numpy()

        # The kernel emits columns in score order, so gather the reference onto
        # the same permutation before comparing.
        cols_seen = seen["columns"][0]
        got = seen["probs"][0]
        valid = cols_seen >= 0
        safe = np.where(valid, cols_seen, 0)
        ref_perm = np.take_along_axis(ref[0], safe, axis=-1)
        ref_perm = np.where(valid, ref_perm, 0.0)
        max_abs = float(np.abs(got - ref_perm).max())
        # bf16 end to end (production dtype), so the reference rebuilt from the
        # rounded weights lands ~2.3e-5 away; the audit's fp32 run gets 3.0e-8.
        # The wrong scale is 7.5e-1, i.e. this threshold still discriminates by
        # nearly three orders of magnitude.
        self.assertLess(
            max_abs,
            1e-3,
            f"warmup probs do not match the reference scale: max|d|={max_abs:.3e}",
        )

    def test_warmup_scores_every_causal_column_via_tilelang(self):
        """Phase 2 scores the whole causal span, in one tilelang call.

        Two things are pinned. First, the **cuDNN** top-k kernel -- phase 3's
        selector -- is called zero times: this phase reads no ``index_topk``, no
        window and no clamped candidate range. Second, the tilelang indexer is
        called exactly once at ``topk_effective == s``, its documented
        "full-candidate selection" mode, and the columns it comes back with are
        exactly the attention table's, diagonal included -- the very column the
        old clamped candidate range could never return.

        Before the phase split this test demanded one *cuDNN* call for a widened
        loss table. That widening was the bug: at the production
        ``index_topk=2048`` it capped back to the phase-3 width, so the KL scored
        the same columns attention would have picked.
        """
        import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as fwd_mod
        import paddlefleet.tilelang_ops as tl_mod

        cudnn_calls = []
        tl_widths = []
        tl_columns = []
        inner_cudnn = fwd_mod.cudnn_indexer_topk_fwd
        inner_tl = tl_mod.csa_indexer_topk_fwd

        def recording_cudnn(*args, **kwargs):
            cudnn_calls.append(int(kwargs["topk_effective"]))
            return inner_cudnn(*args, **kwargs)

        def recording_tl(*args, **kwargs):
            tl_widths.append(int(kwargs["topk_effective"]))
            columns, probs = inner_tl(*args, **kwargs)
            tl_columns.append(columns.numpy().copy())
            return columns, probs

        query, key, w_v, x, qr = self._inputs()
        tensors = [t.clone() for t in (query, key, x, qr)]
        fwd_mod.cudnn_indexer_topk_fwd = recording_cudnn
        tl_mod.csa_indexer_topk_fwd = recording_tl
        try:
            out = self._call(
                self.module, tensors, w_v, training=True, differentiable=True
            )
            out.cast("float32").sum().backward()
        finally:
            fwd_mod.cudnn_indexer_topk_fwd = inner_cudnn
            tl_mod.csa_indexer_topk_fwd = inner_tl

        self.assertEqual(
            cudnn_calls, [], "warmup called the cuDNN indexer top-k kernel"
        )
        self.assertEqual(tl_widths, [self.SEQLEN])
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)

        attn_table = _CAPTURED[-1]
        np.testing.assert_array_equal(
            attn_table, _full_causal_table(self.LAYOUT, self.SEQLEN)
        )
        kl_columns = tl_columns[0]
        for row in range(self.SEQLEN):
            attn_cols = attn_table[0, row]
            kl_cols = kl_columns[0, row]
            self.assertEqual(
                set(kl_cols[kl_cols >= 0].tolist()),
                set(attn_cols[attn_cols >= 0].tolist()),
                f"row {row}: KL and attention column sets differ",
            )
        last = self.SEQLEN - 1
        self.assertIn(last, set(attn_table[0, last].tolist()))
        self.assertIn(last, set(kl_columns[0, last].tolist()))

    def test_eval_early_exit_matches_the_training_forward(self):
        """``:495`` skips the indexer projections entirely under ``eval()``.

        Attention does not consume the indexer in this phase, so with nothing to
        learn this step there is nothing to compute -- and the attention output
        must be bit-identical to the training forward, which does run them.
        """
        query, key, w_v, x, qr = self._inputs(seed=2)
        calls = []
        inner = self.module.indexer.forward_before_topk

        def recording(*args, **kwargs):
            calls.append(len(calls))
            return inner(*args, **kwargs)

        self.module.indexer.forward_before_topk = recording

        tensors = [t.clone() for t in (query, key, x, qr)]
        out_train = _fp32(
            self._call(
                self.module, tensors, w_v, training=True, differentiable=True
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)

        DSAIndexerLossLoggingHelper.tracker.clear()
        out_eval = _fp32(
            self._call(self.module, (query, key, x, qr), w_v, training=False)
        )
        self.assertEqual(len(calls), 1, "eval must not run the indexer at all")
        self.assertNotIn("values", DSAIndexerLossLoggingHelper.tracker)
        np.testing.assert_array_equal(out_train, out_eval)

    def test_indexer_gradients_flow_in_the_warmup_phase(self):
        """All five indexer parameters keep a finite non-zero gradient.

        Phase 2 is where the indexer does all of its learning (the backbone is
        frozen by the trainer), so a silently gradient-free indexer parameter
        would waste the entire phase. Same contract as
        ``TestMQADSA.test_backward_produces_finite_grads_and_reports_loss``,
        extended to ``k_norm`` and driven with attention detached from the
        indexer's output.
        """
        query, key, w_v, x, qr = self._inputs()
        tensors = [query, key, x, qr]
        out = self._call(
            self.module, tensors, w_v, training=True, differentiable=True
        )
        out.cast("float32").sum().backward()
        indexer = self.module.indexer
        params = {
            "wq_b.weight": indexer.wq_b.linear.weight,
            "wk.weight": indexer.wk.linear.weight,
            "k_norm.weight": indexer.k_norm.weight,
            "k_norm.bias": indexer.k_norm.bias,
            "weights_proj.weight": indexer.weights_proj.linear.weight,
        }
        for name, param in params.items():
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
        # Unchanged contract: the indexer learns from its own KL only, so its
        # inputs stay detached while the backbone query/key still flow.
        self.assertIsNone(x.grad)
        self.assertIsNone(qr.grad)
        self.assertIsNotNone(query.grad)
        self.assertTrue(bool(paddle.isfinite(query.grad.cast("float32")).all()))
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)

    def test_recompute_double_forward_table_is_bit_identical(self):
        """Stronger than the phase-3 equivalent, and on the harder layout.

        ``TestMQADSA.test_recompute_double_forward_is_consistent`` has to pick a
        two-document layout because the top-k kernel's emitted order drifts on a
        single full-length document. Phase 2's table contains no floating-point
        scoring at all, so it is bit-identical across the two passes *and* equal
        to the analytic table -- assert both, on the single-document layout that
        the phase-3 path cannot use.
        """
        seqlen = self.SEQLEN
        self.row_end = _row_end([seqlen], seqlen)
        query, key, w_v, x, qr = self._inputs()
        query.stop_gradient = False
        expected = _full_causal_table([seqlen], seqlen)

        _CAPTURED.clear()
        with paddle.no_grad():
            self._call(self.module, (query, key, x, qr), w_v, training=True)
        first = _CAPTURED[-1]
        self.assertNotIn(
            "values",
            DSAIndexerLossLoggingHelper.tracker,
            "indexer loss must not be attached on the no_grad pass",
        )

        self._call(self.module, (query, key, x, qr), w_v, training=True)
        second = _CAPTURED[-1]
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, expected)
        self.assertIn("values", DSAIndexerLossLoggingHelper.tracker)


class TestHashableTensor(unittest.TestCase):
    """``_HashableTensor`` exists only to make the kernel cache key hashable.

    The cuDNN score-recompute wrapper hashes ``(dtype, shape, stride(), ...)``;
    Paddle returns both as lists, which ``dict`` rejects. No GPU needed -- the
    contract is purely about the container types.
    """

    def test_shape_and_stride_are_hashable_tuples(self):
        tensor = _HashableTensor(paddle.zeros([2, 3, 4], dtype="float32"))
        self.assertIsInstance(tensor.shape, tuple)
        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertIsInstance(tensor.stride(), tuple)
        self.assertEqual(tensor.stride(), (12, 4, 1))
        # Per-dim form: the wrapper does not use it, but it is the half of the
        # override that would silently return a plain int if dropped.
        self.assertEqual(tensor.stride(0), 12)
        hash((tensor.shape, tensor.stride()))


@_GPU
class TestMQADSACudnnTarget(unittest.TestCase):
    """The fused indexer-loss target (``_attn_target_cudnn``).

    The kernel needs an LSE taken over exactly the scored column set, which
    exists only when attention and loss share one table (phase 3,
    ``dsa_indexer_use_sparse_loss=True``) and the budget is a width
    ``flash_mla_sparse_fwd`` implements (``_LSE_INDEXER_TOPKS``). The
    module-wide fixture runs ``INDEX_TOPK = 128``, which is not one of them, so
    every test here raises the budget to 512 -- the narrowest supported width.
    """

    TOPK = 512
    SEQLEN = 768  # > WINDOW + TOPK, so the table stays genuinely sparse

    def _build(self, topk):
        config = _create_mqa_config("mqa_dsa", loss_coeff=0.01)
        config.dsa_index_topk = topk
        module = _build_module(config, bf16=True)
        self.assertEqual(int(module.indexer.index_topk), topk)
        return module

    def setUp(self):
        _CAPTURED.clear()
        DSAIndexerLossLoggingHelper.tracker.clear()
        self.module = self._build(self.TOPK)

    def _train_step(self, module, seed=0):
        """One grad-enabled forward + backward, capturing the target's inputs.

        Returns the ``_attn_target`` arguments so a test can re-run the Python
        reference on the *same* columns; recomputing them from a second forward
        would not work, the top-k table drifts between identical calls on a
        single full-length document (``TestMQADSA``
        ``test_use_sparse_loss_switches_both_attention_and_loss_width``).
        """
        query, key, w_v, x, qr = _make_inputs(
            self.SEQLEN, seed=seed, with_hidden=True
        )
        captured = {}
        inner = module._attn_target

        def spy(query_, kv_, topk_indices, lse_indexer=None):
            captured["args"] = (query_, kv_, topk_indices)
            captured["lse_indexer"] = lse_indexer
            target = inner(query_, kv_, topk_indices, lse_indexer)
            captured["target"] = target
            return target

        module._attn_target = spy
        try:
            tensors = [t.clone() for t in (query, key, x, qr)]
            for tensor in tensors:
                tensor.stop_gradient = False
            module.train()
            out = module(
                tensors[0],
                tensors[1],
                None,
                None,
                _row_end([self.SEQLEN], self.SEQLEN),
                v_b_proj_weight=w_v,
                x=tensors[2],
                qr=tensors[3],
            )
            out.cast("float32").sum().backward()
        finally:
            module._attn_target = inner
        return captured

    def test_supported_budget_takes_the_fused_path(self):
        """A 512 budget routes the target through the kernel, and the KL lands.

        Asserts the plumbing end to end: ``mqa_sparse_attn`` returns the second
        output at all (the ``indexer_topk > 0`` signature), the LSE has the
        kernel's fixed ``h_q == 64`` head count rather than the layer's 8, and
        the loss built on top of it is finite and non-zero.

        The LSE is finite exactly on the rows that have a candidate: it is a
        log-sum-exp over the ``indexer_topk`` prefix of the table, so the rows
        whose prefix is all ``-1`` -- the first ``window_size`` tokens of a
        document, which ``_indexer_valid_range`` leaves without candidates --
        come back ``+inf``. Those rows have to end up as a zero target row, not
        as a NaN one, which is what the KL's valid-row denominator assumes.
        """
        captured = self._train_step(self.module)
        lse = captured["lse_indexer"]
        self.assertIsNotNone(lse, "the fused path did not receive an LSE")
        self.assertEqual(list(lse.shape), [1, self.SEQLEN, 64])
        self.assertEqual(lse.dtype, paddle.float32)
        has_candidate = (captured["args"][2] >= 0).any(axis=-1)
        self.assertTrue(
            bool(has_candidate.any()) and not bool(has_candidate.all())
        )
        self.assertTrue(bool(paddle.isfinite(lse[has_candidate]).all()))
        self.assertTrue(bool(paddle.isinf(lse[~has_candidate]).all()))

        target = captured["target"]
        self.assertEqual(list(target.shape), [1, self.SEQLEN, self.TOPK])
        self.assertTrue(bool(paddle.isfinite(target).all()))
        self.assertEqual(float(target[~has_candidate].abs().max()), 0.0)
        np.testing.assert_allclose(
            target.sum(axis=-1)[has_candidate].numpy(), 1.0, atol=1e-5
        )
        loss = float(DSAIndexerLossLoggingHelper.tracker["values"][0])
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_fused_target_matches_the_python_reference(self):
        """The kernel and ``_attn_target_python`` must agree on the same table.

        Both produce ``sum_h softmax_h`` over the selected columns, L1
        normalised, so this is the correctness contract of the whole change: the
        head-sum's normalizer is the LSE restricted to those columns, and
        getting that wrong (e.g. feeding the attention LSE, which also covers
        the window and the sink) changes the objective without changing the
        forward output.
        """
        captured = self._train_step(self.module)
        query, kv, topk_indices = captured["args"]
        fused = captured["target"]
        reference = self.module._attn_target_python(query, kv, topk_indices)

        self.assertLess(_rel(fused, reference), 5e-3)
        # Every non-empty row is a distribution; empty slots stay exactly zero
        # (the KL divides by the valid-row count, not by the row sum).
        valid_rows = (topk_indices >= 0).any(axis=-1)
        row_sums = fused.sum(axis=-1)
        np.testing.assert_allclose(row_sums[valid_rows].numpy(), 1.0, atol=1e-5)
        empty_slots = topk_indices < 0
        self.assertEqual(float(fused[empty_slots].abs().max()), 0.0)

    def test_unsupported_budget_falls_back_to_python(self):
        """384 is a legal ``index_topk`` the kernel does not implement.

        ``dsa_index_topk`` only has to be a multiple of 128 and at most 2048
        (``transformer_config.py``), while ``indexer_topk`` accepts
        0/512/1024/2048 only. Without the ``_LSE_INDEXER_TOPKS`` guard the
        illegal width would reach the kernel; with it the target silently uses
        the Python reference instead.
        """
        self.assertNotIn(384, _LSE_INDEXER_TOPKS)
        module = self._build(384)
        captured = self._train_step(module)
        self.assertIsNone(captured["lse_indexer"])
        loss = float(DSAIndexerLossLoggingHelper.tracker["values"][0])
        self.assertTrue(np.isfinite(loss))


def _fake_score_recompute(fn):
    """Put ``fn`` in place of the cuDNN score-recompute wrapper.

    ``_attn_target_cudnn`` imports it lazily from
    ``paddlefleet_ops.cudnn.deepseek_sparse_attention``, so swapping the module
    in ``sys.modules`` intercepts the call without importing (or owning a GPU
    able to run) the real op.
    """
    module = types.ModuleType("paddlefleet_ops.cudnn.deepseek_sparse_attention")
    module.sparse_attn_score_recompute_wrapper = fn
    return mock.patch.dict(
        sys.modules,
        {"paddlefleet_ops.cudnn.deepseek_sparse_attention": module},
    )


class TestAttnTargetDispatch(unittest.TestCase):
    """``_attn_target`` picks its implementation from ``lse_indexer`` alone.

    No GPU: the dispatch is what decides whether the loss target comes from the
    kernel or from the reference, and it must not consult anything else (the
    caller has already resolved the phase and the budget).
    """

    def setUp(self):
        self.calls = []
        self.stub = SimpleNamespace(
            _attn_target_cudnn=lambda *a: self.calls.append("cudnn") or "cudnn",
            _attn_target_python=lambda *a: self.calls.append("python")
            or "python",
        )

    def test_lse_present_selects_the_kernel(self):
        got = MQALatentAttention._attn_target(
            self.stub, "q", "kv", "idx", "lse"
        )
        self.assertEqual((got, self.calls), ("cudnn", ["cudnn"]))

    def test_lse_absent_selects_the_reference(self):
        got = MQALatentAttention._attn_target(self.stub, "q", "kv", "idx", None)
        self.assertEqual((got, self.calls), ("python", ["python"]))
        # The default is the fallback too: phase 2 calls it with three args.
        self.assertEqual(
            MQALatentAttention._attn_target(self.stub, "q", "kv", "idx"),
            "python",
        )


class TestAttnTargetCudnnMocked(unittest.TestCase):
    """``_attn_target_cudnn``'s call contract, kernel mocked out.

    The kernel itself needs SM100+ (covered by ``TestMQADSACudnnTarget``); what
    is checked here is everything the wrapper is responsible for and the kernel
    is not: hashable cache-key metadata, the LSE sliced down to the real head
    count and then both it and the query padded back up to the kernel's
    narrowest MMA tile, int32 indices, and the empty-slot handling on the way
    out.
    """

    H_Q, S, TOPK, DK_ = 4, 3, 4, 8
    SCALE = 0.25

    def _inputs(self, h_q=None):
        h_q = self.H_Q if h_q is None else h_q
        query = paddle.zeros([1, self.S, h_q, self.DK_], dtype="bfloat16")
        kv = paddle.zeros([1, self.S, self.DK_], dtype="bfloat16")
        # Row 0 has two valid columns, row 1 one, row 2 none (a fully padded
        # query row, which a short document's first token produces).
        idx = paddle.to_tensor(
            [[[0, 1, -1, -1], [0, -1, -1, -1], [-1, -1, -1, -1]]],
            dtype="int64",
        )
        # The kernel's LSE always has the DSA-fixed 64 heads, not the layer's.
        lse = paddle.full([1, self.S, 64], 1.5, dtype="bfloat16")
        return query, kv, idx, lse

    def _run(self, target_value=2.0, h_q=None):
        seen = {}

        def fake(q, kv, lse, idx, scale):
            seen["types"] = [type(t) for t in (q, kv, lse, idx)]
            # The real wrapper keys its kernel cache on this tuple; lists would
            # raise TypeError here, which is the whole point of
            # ``_HashableTensor``.
            seen["key"] = hash(
                tuple((t.dtype, t.shape, t.stride()) for t in (q, kv, lse, idx))
            )
            seen["q_shape"] = list(q.shape)
            seen["q"] = q.cast("float32").numpy().copy()
            seen["lse_shape"] = list(lse.shape)
            seen["lse"] = lse.numpy().copy()
            seen["lse_dtype"] = lse.dtype
            seen["idx_dtype"] = idx.dtype
            seen["scale"] = scale
            return {
                "target": paddle.full(idx.shape, target_value, dtype="float32")
            }

        query, kv, idx, lse = self._inputs(h_q)
        with _fake_score_recompute(fake):
            target = MQALatentAttention._attn_target_cudnn(
                SimpleNamespace(softmax_scale=self.SCALE), query, kv, idx, lse
            )
        return seen, target

    def test_kernel_arguments(self):
        seen, _ = self._run()
        self.assertEqual(seen["types"], [_HashableTensor] * 4)
        self.assertEqual(seen["lse_dtype"], paddle.float32)
        self.assertEqual(seen["idx_dtype"], paddle.int32)
        self.assertEqual(seen["scale"], self.SCALE)

    def test_narrow_head_count_is_padded_with_an_infinite_lse(self):
        """``h < 16`` is padded up, and the pad heads contribute nothing.

        The kernel's MMA ``M`` tile is the query-head count and it silently
        returns an all-zero target below 16 heads, so the wrapper pads. The pad
        heads must not join the head sum, which an infinite LSE guarantees
        exactly: ``exp(finite - inf) == 0``.
        """
        seen, _ = self._run()
        self.assertEqual(seen["q_shape"], [1, self.S, 16, self.DK_])
        self.assertEqual(seen["lse_shape"], [1, self.S, 16])
        # Real heads keep the layer's LSE, sliced out of the kernel's 64-wide
        # one; the pad heads are +inf.
        np.testing.assert_array_equal(seen["lse"][:, :, : self.H_Q], 1.5)
        self.assertTrue(bool(np.isposinf(seen["lse"][:, :, self.H_Q :]).all()))
        np.testing.assert_array_equal(seen["q"][:, :, self.H_Q :], 0.0)

    def test_supported_head_count_is_passed_through(self):
        """A power-of-two ``h >= 16`` (production is 64) is not padded."""
        seen, _ = self._run(h_q=64)
        self.assertEqual(seen["q_shape"], [1, self.S, 64, self.DK_])
        self.assertEqual(seen["lse_shape"], [1, self.S, 64])
        self.assertTrue(bool(np.isfinite(seen["lse"]).all()))

    def test_empty_slots_are_zeroed_and_rows_renormalised(self):
        """Whatever the kernel writes at ``-1`` slots is discarded.

        A uniform kernel output makes the expectation exact: each row becomes
        uniform over its valid columns, and the all-empty row stays all zeros
        rather than turning into ``0/0`` -- the KL reduction divides by the
        valid-row count, so a padded row must contribute nothing.
        """
        _, target = self._run()
        np.testing.assert_allclose(
            target.numpy(),
            np.array([[[0.5, 0.5, 0.0, 0.0], [1.0, 0, 0, 0], [0, 0, 0, 0]]]),
            atol=1e-6,
        )
        self.assertEqual(target.dtype, paddle.float32)


class TestMQASparseAttnLseSideChannelMocked(unittest.TestCase):
    """``mqa_sparse_attn``'s ``lse_indexer`` side channel, kernel mocked out.

    The LSE cannot be a PyLayer output (every returned tensor would demand a
    matching backward gradient), so it travels on a class attribute that the
    wrapper pops. That popping is pure Python and is what this checks, together
    with the ``indexer_topk`` pass-through and the two return signatures.
    """

    S, H_Q, DK_, DV_ = 4, 8, 576, 512
    SCALE = 0.1

    def _inputs(self):
        query = paddle.zeros([1, self.S, self.H_Q, self.DK_], dtype="bfloat16")
        kv = paddle.zeros([1, self.S, self.DK_], dtype="bfloat16")
        idx = paddle.arange(self.S, dtype="int32").reshape([1, 1, self.S])
        idx = paddle.where(
            idx <= paddle.arange(self.S, dtype="int32").reshape([1, self.S, 1]),
            idx.tile([1, self.S, 1]),
            paddle.full([1, self.S, self.S], -1, dtype="int32"),
        )
        return query, kv, idx.contiguous()

    def _patched_kernel(self, calls):
        import paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn as fwd_mod

        def fake(q_pad, kv, sink, token_indices, **kwargs):
            calls.append(kwargs)
            b, s = int(q_pad.shape[0]), int(q_pad.shape[1])
            out = paddle.zeros([b, s, 64, kwargs["d_v"]], dtype=q_pad.dtype)
            lse = paddle.zeros([b, s, 64], dtype="float32")
            # The kernel returns ``None`` for a 0 budget; a non-None value for
            # a real one, so a leak would be visible on the next call.
            lse_indexer = (
                None
                if kwargs["indexer_topk"] == 0
                else paddle.full([b, s, 64], 1.5, dtype="float32")
            )
            return out, lse, lse_indexer

        return mock.patch.object(fwd_mod, "flash_mla_sparse_attn", fake)

    def _call(self, indexer_topk, calls):
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        query, kv, idx = self._inputs()
        with self._patched_kernel(calls):
            return mqa_sparse_attn(
                query,
                kv,
                idx,
                self.SCALE,
                self.DV_,
                attn_sink=None,
                indexer_topk=indexer_topk,
            )

    def tearDown(self):
        from paddlefleet.fusions.mqa_sparse_attn import _MQASparseAttention

        _MQASparseAttention._lse_indexer = None

    def test_zero_budget_keeps_the_single_output_signature(self):
        calls = []
        out = self._call(0, calls)
        self.assertIsInstance(out, paddle.Tensor)
        self.assertEqual(list(out.shape), [1, self.S, self.H_Q * self.DV_])
        self.assertEqual(calls[0]["indexer_topk"], 0)

    def test_positive_budget_returns_and_forwards_the_lse(self):
        calls = []
        out, lse_indexer = self._call(512, calls)
        self.assertEqual(calls[0]["indexer_topk"], 512)
        self.assertEqual(list(out.shape), [1, self.S, self.H_Q * self.DV_])
        self.assertEqual(list(lse_indexer.shape), [1, self.S, 64])
        self.assertEqual(lse_indexer.dtype, paddle.float32)

    def test_the_side_channel_never_outlives_one_call(self):
        """Popped on every call, so a stale LSE cannot reach the next one.

        Without the reset a 0-budget call following a 512 one would still find
        the previous tensor on the class -- harmless today (the return value is
        gated on ``indexer_topk``) but it would pin one LSE per layer alive for
        the whole step.
        """
        from paddlefleet.fusions.mqa_sparse_attn import _MQASparseAttention

        calls = []
        self._call(512, calls)
        self.assertIsNone(_MQASparseAttention._lse_indexer)
        self._call(0, calls)
        self.assertIsNone(_MQASparseAttention._lse_indexer)


class TestSparseAttnPlumbingMocked(unittest.TestCase):
    """``_sparse_attn`` forwards the sink and the budget, and nothing else.

    Mocking ``mqa_sparse_attn`` keeps this off the SM100 kernels; the method's
    only job is to hand over ``self.softmax_offset`` and ``indexer_topk``.
    """

    def setUp(self):
        _CAPTURED.clear()
        self.module = _build_module(_create_mqa_config("mqa_dsa"), bf16=True)

    def _patched(self, calls):
        import paddlefleet.fusions.mqa_sparse_attn as fusion

        def fake(query, kv, token_indices, sm_scale, d_v, **kwargs):
            calls.append((float(sm_scale), int(d_v), kwargs))
            return "core_out"

        return mock.patch.object(fusion, "mqa_sparse_attn", fake)

    def test_budget_and_sink_are_forwarded(self):
        calls = []
        with self._patched(calls):
            got = self.module._sparse_attn(
                paddle.zeros([1, 2, H, DK], dtype="bfloat16"),
                paddle.zeros([1, 2, DK], dtype="bfloat16"),
                paddle.zeros([1, 2, 4], dtype="int32"),
                self.module.softmax_scale,
                DV,
                indexer_topk=512,
            )
        self.assertEqual(got, "core_out")
        scale, d_v, kwargs = calls[0]
        self.assertEqual((scale, d_v), (self.module.softmax_scale, DV))
        self.assertEqual(kwargs["indexer_topk"], 512)
        # No sink configured in this fixture: the backend reads ``None`` as
        # "sinkless softmax", it is not an omitted argument.
        self.assertIsNone(kwargs["attn_sink"])
        self.assertIn("attn_sink", kwargs)


class TestForwardDsaFusedDispatchMocked(unittest.TestCase):
    """Which target path ``_forward_sparse`` selects, with both kernels mocked.

    ``TestMQADSACudnnTarget`` covers the same decision on real kernels; this
    reaches it without any, so the branch stays checked on machines below
    SM100. Mocked: the indexer top-k kernel, the sparse-attention call and the
    target itself -- everything between them (window table, concat order, KL,
    loss logging) is the real code.
    """

    S = 256
    WIDE_TOPK = 384  # a legal index_topk the kernel does not implement

    def _build(self, topk):
        config = _create_mqa_config("mqa_dsa", loss_coeff=0.01)
        config.dsa_index_topk = topk
        module = _build_module(config, bf16=True)
        module.train()
        return module

    def _topk_table(self, topk):
        """``[1, S, topk]`` causal-ish table, right-padded with ``-1``."""
        cols = paddle.arange(topk, dtype="int32").reshape([1, 1, topk])
        rows = paddle.arange(self.S, dtype="int32").reshape([1, self.S, 1])
        return paddle.where(
            cols <= rows,
            cols.tile([1, self.S, 1]),
            paddle.full([1, self.S, topk], -1, dtype="int32"),
        ).contiguous()

    def _run(self, topk):
        module = self._build(topk)
        DSAIndexerLossLoggingHelper.tracker.clear()
        seen = {}
        table = self._topk_table(topk)

        import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as fwd_mod

        def fake_topk(q, k, w, **kwargs):
            width = int(kwargs["topk_effective"])
            scores = paddle.rand([1, self.S, width], dtype="float32")
            return self._topk_table(width), None, scores

        def fake_indexer(x, qr, position_offset, cp_group):
            return (
                paddle.zeros(
                    [1, self.S, INDEX_HEADS, INDEX_HEAD_DIM], dtype="bfloat16"
                ),
                paddle.zeros([1, self.S, INDEX_HEAD_DIM], dtype="bfloat16"),
                paddle.zeros([1, self.S, INDEX_HEADS], dtype="bfloat16"),
            )

        def fake_sparse_attn(
            query, kv, token_indices, sm_scale, d_v, indexer_topk=0
        ):
            seen["indexer_topk"] = int(indexer_topk)
            seen["token_indices"] = token_indices.numpy().copy()
            core_out = query.reshape([1, self.S, H * DK])[:, :, : H * DV]
            if indexer_topk == 0:
                return core_out
            return core_out, paddle.full([1, self.S, 64], 1.5, dtype="float32")

        def fake_target(query, kv, topk_indices, lse_indexer=None):
            seen["lse_indexer"] = lse_indexer
            width = int(topk_indices.shape[-1])
            return paddle.full([1, self.S, width], 1.0 / width, dtype="float32")

        query, key, w_v, x, qr = _make_inputs(self.S, seed=0, with_hidden=True)
        tensors = [t.clone() for t in (query, key, x, qr)]
        for tensor in tensors:
            tensor.stop_gradient = False
        module.indexer.forward_before_topk = fake_indexer
        module._sparse_attn = fake_sparse_attn
        module._attn_target = fake_target
        with mock.patch.object(fwd_mod, "cudnn_indexer_topk_fwd", fake_topk):
            out = module(
                tensors[0],
                tensors[1],
                None,
                None,
                _row_end([self.S], self.S),
                v_b_proj_weight=w_v,
                x=tensors[2],
                qr=tensors[3],
            )
        seen["output"] = out
        seen["table"] = table.numpy()
        return seen

    def test_supported_budget_requests_the_lse_and_reuses_it(self):
        topk = _LSE_INDEXER_TOPKS[0]
        seen = self._run(topk)
        self.assertEqual(seen["indexer_topk"], topk)
        # The indexer columns must come first: the kernel's LSE covers the
        # leading ``indexer_topk`` columns of the table, so the window has to
        # sit in the tail.
        table = seen["token_indices"]
        self.assertEqual(table.shape[-1], topk + WINDOW)
        doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(
            _row_end([self.S], self.S), self.S
        )
        window = (
            _build_window_topk_idxs_from_doc_bounds(
                1, self.S, WINDOW, doc_start, is_valid
            )
            .cast("int32")
            .numpy()
        )
        np.testing.assert_array_equal(table[:, :, topk:], window)
        # ``_forward_sparse`` blanks the rows whose candidate range is empty (the
        # first ``window_size`` tokens of a document), so compare where the
        # prefix survived.
        prefix, selected = table[:, :, :topk], seen["table"]
        kept = prefix != -1
        self.assertGreater(int(kept.sum()), 0)
        np.testing.assert_array_equal(prefix[kept], selected[kept])
        self.assertIsNotNone(seen["lse_indexer"])
        self.assertEqual(list(seen["lse_indexer"].shape), [1, self.S, 64])
        loss = float(DSAIndexerLossLoggingHelper.tracker["values"][0])
        self.assertTrue(np.isfinite(loss))

    def test_unsupported_budget_asks_for_no_lse(self):
        self.assertNotIn(self.WIDE_TOPK, _LSE_INDEXER_TOPKS)
        seen = self._run(self.WIDE_TOPK)
        self.assertEqual(seen["indexer_topk"], 0)
        self.assertIsNone(seen["lse_indexer"])
        loss = float(DSAIndexerLossLoggingHelper.tracker["values"][0])
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
