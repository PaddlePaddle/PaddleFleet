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
"""

import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import paddle

from paddlefleet.transformer.csa_attention import (
    _build_window_topk_idxs_from_doc_bounds,
    _derive_csa_doc_boundaries,
)
from paddlefleet.transformer.dsa_attention import DSAIndexerLossLoggingHelper
from paddlefleet.transformer.mqa_latent_attention import MQALatentAttention
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


def _full_causal_table(layout, seqlen):
    """The per-document full-causal ``[1, s, s]`` table, from the production
    builder itself -- it is a pure integer function of the document bounds.
    """
    row_end = _row_end(layout, seqlen)
    doc_start, _, is_valid, _, _ = _derive_csa_doc_boundaries(row_end, seqlen)
    table = MQALatentAttention._build_full_causal_indices(
        1, seqlen, doc_start, is_valid
    )
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

    def test_softmax_scale_is_the_mha_scale(self):
        # Absorption is exactly score preserving, so the scale must stay the MHA
        # q_head_dim one (256**-0.5), never the 576-wide latent one.
        self.assertAlmostEqual(
            self.module.softmax_scale, K_CHANNELS**-0.5, places=12
        )
        self.assertAlmostEqual(self.module.softmax_scale, 0.0625, places=12)

    def test_pad_token_id_none_rejected(self):
        # A ``None`` ``pad_token_id`` would compare every token against None,
        # marking the whole batch valid and silently changing the indexer loss
        # denominator. Must raise, and must not be an ``assert`` (see
        # ``TestValidationSurvivesOptimizedMode``).
        fake = SimpleNamespace(
            config=SimpleNamespace(pad_token_id=None), cp_enabled=False
        )
        with self.assertRaisesRegex(ValueError, "pad_token_id must be set"):
            MQALatentAttention._indexer_loss_mask(
                fake, paddle.zeros([1, 4], dtype="int64"), 1, 4
            )

    def test_attn_sink_wrong_head_count_rejected(self):
        # The sparse kernel fixes ``h_q`` at 64 and zero-pads shorter head
        # dims, so a wrong-length sink is not caught downstream -- it would be
        # padded and consumed as if valid. The check fires before any kernel
        # launch, so no SM10.x device is required.
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        with self.assertRaisesRegex(ValueError, "attn_sink must be"):
            mqa_sparse_attn(
                paddle.zeros([1, 2, 4, DK]),
                paddle.zeros([1, 2, DK]),
                paddle.zeros([1, 2, 4], dtype="int32"),
                0.0625,
                DV,
                attn_sink=paddle.zeros([5]),
            )


class TestValidationSurvivesOptimizedMode(unittest.TestCase):
    """The two public-entry validations must be real ``raise`` statements.

    ``python -O`` strips ``assert``, so an ``assert``-based check is no check at
    all in an optimized deployment: a wrong-length ``attn_sink`` would be
    zero-padded into the fixed 64-head CUDA kernel and a ``None``
    ``pad_token_id`` would mark every row valid. Both checks fire before any
    kernel launch, so no SM10.x device is required.
    """

    _SCRIPT = """
import sys
from types import SimpleNamespace

import paddle

if sys.flags.optimize < 1:
    print("NOT_OPTIMIZED")
    sys.exit(3)

from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn
from paddlefleet.transformer.mqa_latent_attention import MQALatentAttention

try:
    mqa_sparse_attn(
        paddle.zeros([1, 2, 4, 576]),
        paddle.zeros([1, 2, 576]),
        paddle.zeros([1, 2, 4], dtype="int32"),
        0.0625,
        512,
        attn_sink=paddle.zeros([5]),
    )
except ValueError as exc:
    if "attn_sink must be" not in str(exc):
        print("BAD_SINK_MESSAGE", exc)
        sys.exit(4)
else:
    print("SINK_CHECK_STRIPPED")
    sys.exit(5)

fake = SimpleNamespace(
    config=SimpleNamespace(pad_token_id=None), cp_enabled=False
)
try:
    MQALatentAttention._indexer_loss_mask(
        fake, paddle.zeros([1, 4], dtype="int64"), 1, 4
    )
except ValueError as exc:
    if "pad_token_id must be set" not in str(exc):
        print("BAD_PAD_MESSAGE", exc)
        sys.exit(6)
else:
    print("PAD_CHECK_STRIPPED")
    sys.exit(7)

print("OK")
"""

    def test_both_validations_still_raise(self):
        proc = subprocess.run(
            [sys.executable, "-O", "-c", self._SCRIPT],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}",
        )
        self.assertIn("OK", proc.stdout)


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

    def test_indexer_init_from_scratch_is_rejected_for_the_sparse_phase(self):
        """A from-scratch Indexer is only legal for the warmup phase.

        ``indexer_init_from_scratch=True`` makes ``_gen_aoa_config`` emit the
        ``_ -> key`` add primitive, and that primitive ignores a checkpoint
        tensor of the same name rather than preferring it
        (``aoa_engine.py:581-586`` routes it into ``need_add_output_vars``,
        ``:685-686`` sets ``output_vars[key] = None``, ``:715-720`` returns no
        source slices). The sparse phase always resumes a warmup checkpoint,
        which does hold trained Indexer weights, so the pair would throw the
        whole warmup phase away and report nothing louder than an "Unexpected
        keys" warning. Rejecting it in ``__post_init__`` is the only place that
        can fail *before* a weight is loaded.

        Both fields are shared by the two Indexer flavours, so the rejection is
        flavour-agnostic: it fires for the ``DSAIndexer`` of a ``-2`` layer
        under ``"mqa_dsa"`` *and* for the ``CSAIndexer`` of a ``1 < ratio <
        128`` layer, which exists for any ``hybrid_mla_attention`` as long as
        ``csa_dense_mode=False``. A config that builds neither is not policed,
        otherwise this would just be forbidding the field.
        """
        with self.assertRaisesRegex(ValueError, "is not a valid phase"):
            TransformerConfig(
                **self._mqa_dsa_kwargs(
                    dsa_indexer_use_sparse_loss=True,
                    indexer_init_from_scratch=True,
                )
            )
        legal = (
            # warmup: the phase-1 checkpoint has no Indexer at all
            (False, True),
            # sparse phase resuming a warmup checkpoint that has one
            (True, False),
        )
        for sparse_loss, scratch in legal:
            with self.subTest(sparse_loss=sparse_loss, scratch=scratch):
                config = TransformerConfig(
                    **self._mqa_dsa_kwargs(
                        dsa_indexer_use_sparse_loss=sparse_loss,
                        indexer_init_from_scratch=scratch,
                        dsa_indexer_loss_coeff=0.01,
                    )
                )
                self.assertEqual(config.indexer_init_from_scratch, scratch)
                self.assertEqual(
                    config.dsa_indexer_use_sparse_loss, sparse_loss
                )
        # The CSA flavour, i.e. no latent MQA layer and no "mqa_dsa" at all: a
        # ratio in [2, 127] builds a CSAIndexer unless csa_dense_mode drops it,
        # and CSA reads dsa_indexer_use_sparse_loss as the same phase selector
        # (``_resolve_topk_effective``, csa_attention.py:2059-2074). So the same
        # pair discards the same trained weights and must be rejected the same
        # way -- and must stay legal once csa_dense_mode builds no Indexer.
        csa_kwargs = {
            "hybrid_mla_attention": "mha",
            "csa_compress_ratios": [64, 128],
            "dsa_indexer_use_sparse_loss": True,
            "indexer_init_from_scratch": True,
            "dsa_index_n_heads": INDEX_HEADS,
            "dsa_index_head_dim": INDEX_HEAD_DIM,
            "dsa_index_topk": INDEX_TOPK,
        }
        with self.assertRaisesRegex(ValueError, "CSAIndexer=True"):
            TransformerConfig(**self._kwargs(**csa_kwargs))
        config = TransformerConfig(
            **self._kwargs(csa_dense_mode=True, **csa_kwargs)
        )
        self.assertEqual(config.indexer_init_from_scratch, True)
        # ``"mha"`` over -2 layers builds no Indexer either, so neither field
        # means anything there and the pair must not be policed.
        config = TransformerConfig(
            **self._kwargs(
                hybrid_mla_attention="mha",
                dsa_indexer_use_sparse_loss=True,
                indexer_init_from_scratch=True,
            )
        )
        self.assertEqual(config.indexer_init_from_scratch, True)

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

        Both column sets are asserted. The ``False`` attention table is *exactly*
        assertable: it is ``_build_full_causal_indices``, a pure integer function
        of the document bounds with no floating-point scoring in it, so it is
        reproducible and equal to the builder's own output element for element.

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

        def recording_target(query_, kv_, kl_columns):
            # The KL's column set is the indexer's candidate set: the top-k in
            # the sparse phase, every causal column in warmup. The forced window
            # is never in it.
            loss_widths.append(int(kl_columns.shape[-1]))
            return inner_target(query_, kv_, kl_columns)

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

    def test_sink_grad_dtype_follows_the_forward_input(self):
        """``d_sink`` comes back in the dtype the caller passed in.

        The analytic sink gradient is derived in fp32 (the kernels' compute
        dtype), but a PyLayer must hand each input's gradient back in that
        input's dtype -- an fp32 grad on a bf16 parameter breaks accumulation
        and the optimizer's master-weight path. This drives the shared PyLayer
        directly so the assertion is on the layer contract, not on whatever
        paddle's parameter-grad plumbing happens to coerce.
        """
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        b, s, h = 1, WINDOW, H
        rng = np.random.RandomState(0)
        cols = np.full([b, s, WINDOW], -1, dtype=np.int32)
        for i in range(s):
            cols[0, i, : i + 1] = np.arange(i + 1, dtype=np.int32)
        token_indices = paddle.to_tensor(cols)
        for dtype in ("bfloat16", "float32"):
            with self.subTest(sink_dtype=dtype):
                query = paddle.to_tensor(
                    rng.randn(b, s, h, DK).astype("float32") * 0.1
                ).cast("bfloat16")
                kv = paddle.to_tensor(
                    rng.randn(b, s, DK).astype("float32") * 0.1
                ).cast("bfloat16")
                sink = paddle.to_tensor(
                    np.linspace(1.0, 3.0, h).astype("float32")
                ).cast(dtype)
                sink.stop_gradient = False
                out = mqa_sparse_attn(
                    query, kv, token_indices, K_CHANNELS**-0.5, DV, sink
                )
                out.cast("float32").sum().backward()
                self.assertIsNotNone(sink.grad)
                self.assertEqual(sink.grad.dtype, sink.dtype)
                self.assertEqual(list(sink.grad.shape), [h])
                self.assertGreater(
                    float(sink.grad.astype("float32").abs().max()), 0.0
                )


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

    @classmethod
    def setUpClass(cls):
        try:
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        except Exception:
            pass

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

        Both paths call the same ``_build_full_causal_indices`` and then the
        same sparse kernel, so the outputs must be bit-identical, not merely
        close. Nothing needs weight copying: on this path attention consumes no
        module parameter at all -- the query/key/``v_b_proj_weight`` are inputs
        and ``softmax_offset`` is ``None`` in both -- so the only thing that
        could differ is the index table. Asserted in both modes, so the
        ``:495`` early exit and the full ``:600`` branch are each covered.
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


if __name__ == "__main__":
    unittest.main()
