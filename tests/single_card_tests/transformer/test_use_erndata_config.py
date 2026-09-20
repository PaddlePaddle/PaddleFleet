# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TransformerConfig.use_erndata validation branches.

``use_erndata`` selects the MTP data-flow contract: True means the erndata
(Energon) pipeline emits length-L tensors + cu_seqlens_q, False means the
historical ernie5 L+K layout. Guards checked here:
  1. Default is False.
  2. The removed `mtp_num_layers` alias is rejected as a constructor kwarg
     (TypeError) and, via ``from_config``, whenever it is non-zero.
  3. use_erndata + MTP is incompatible with enable_mtp_magic_send.
  4. use_erndata + MTP is incompatible with experimental_dataflow.
  5. use_erndata without MTP (K == 0) trips none of the guards — the packed-doc
     forward is only reachable through the MTP layer.
  6. use_erndata=False never trips any of the guards regardless of other MTP
     flags.
  7. use_erndata + MTP + CP>1 accepts both sequence-scatter layouts
     (``dualchunk_allgather`` / ``contiguous_allgather``) and rejects
     ``contiguous_a2a``.
  8. use_erndata + MTP + CP>1 rejects ``gpt_model_use_experimental_version``
     (2-column mask the CP expansion refuses without experimental_dataflow),
     while CP==1 keeps accepting it.
  9. use_erndata + MTP + CP>1 rejects a model that builds an Indexer, whose
     loss-mask path assumes a CP-local ``input_ids``; configs that build no
     Indexer, and CP==1, stay legal.
 10. use_erndata + MTP + CP>1 rejects ``mtp_distillation_loss`` (that branch
     double-scatters its loss mask).
 11. use_erndata + CP>1 requires one of the two paths that slice the
     full-length tensors the loader broadcasts: a real MTP layer, or
     ``experimental_dataflow`` (whose ContextParallelScatterOp covers the
     no-MTP case).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from paddlefleet.transformer.transformer_config import TransformerConfig


class TestUseErndataValidation(unittest.TestCase):
    """The use_erndata field must be validated in __post_init__."""

    def _base_kwargs(self, **overrides):
        """Minimal kwargs to build a TransformerConfig without noise from
        other fields. All fields have defaults; we only override MTP-related
        ones needed for a given test case."""
        return dict(overrides)

    def test_default_is_false(self) -> None:
        cfg = TransformerConfig(**self._base_kwargs())
        self.assertFalse(cfg.use_erndata)

    def test_erndata_without_mtp_is_accepted(self) -> None:
        # K == 0 means no MTP layer, so the packed-doc forward is unreachable
        # and none of the MTP-specific guards apply.
        cfg = TransformerConfig(
            **self._base_kwargs(
                use_erndata=True,
                num_nextn_predict_layers=0,
                experimental_dataflow=True,
            )
        )
        self.assertTrue(cfg.use_erndata)
        self.assertEqual(cfg.num_nextn_predict_layers, 0)

    def test_erndata_accepts_num_nextn(self) -> None:
        cfg = TransformerConfig(
            **self._base_kwargs(
                use_erndata=True,
                num_nextn_predict_layers=1,
            )
        )
        self.assertTrue(cfg.use_erndata)

    def test_mtp_num_layers_is_rejected(self) -> None:
        # The alias has been removed from TransformerConfig entirely; passing it
        # must fail loudly rather than silently configure nothing.
        with self.assertRaises(TypeError):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=0,
                    mtp_num_layers=2,
                )
            )

    def test_mtp_num_layers_is_rejected_via_from_config(self) -> None:
        # The path a model_config.json actually takes. Without a
        # renamed_config_keys_when_set entry _process_attribute's setattr
        # fallback would absorb the key as a dead attribute and leave MTP
        # silently off.
        cfg_in = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_nextn_predict_layers=0,
            mtp_num_layers=2,
        )
        with self.assertRaisesRegex(
            ValueError, r"num_nextn_predict_layers"
        ) as ctx:
            TransformerConfig.from_config(cfg_in)
        self.assertIn("mtp_num_layers", str(ctx.exception))

    def test_zero_mtp_num_layers_is_tolerated_via_from_config(self) -> None:
        # PaddleFormers declares `mtp_num_layers` itself (LlmMetaConfig /
        # TrainingArguments, default 0), so every config it produces hands the
        # key over even when MTP is off. Rejecting a zero would break every
        # Fleet-provider model in that repo, and it carries no information the
        # key's absence does not.
        cfg_in = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_nextn_predict_layers=1,
            mtp_num_layers=0,
        )
        cfg = TransformerConfig.from_config(cfg_in)
        self.assertEqual(cfg.num_nextn_predict_layers, 1)

    def test_mtp_num_layers_registered_as_renamed_when_set(self) -> None:
        self.assertIn(
            "mtp_num_layers", TransformerConfig.renamed_config_keys_when_set
        )
        self.assertNotIn(
            "mtp_num_layers", TransformerConfig.renamed_config_keys
        )

    def test_erndata_incompat_with_magic_send(self) -> None:
        # enable_mtp_magic_send also requires PP>1 (checked earlier in
        # __post_init__), so we build a config that would pass that check.
        #
        # Assert on the flag name alone, not the prose: the reasoning for the
        # rejection lives in the comment above the guard, and pinning it here
        # would make every rewording a CI failure.
        with self.assertRaisesRegex(ValueError, r"enable_mtp_magic_send"):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=1,
                    enable_mtp_magic_send=True,
                    pipeline_model_parallel_size=2,
                )
            )

    def test_erndata_incompat_with_experimental_dataflow(self) -> None:
        with self.assertRaisesRegex(ValueError, r"experimental_dataflow"):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=1,
                    experimental_dataflow=True,
                )
            )

    def test_erndata_incompat_with_separate_mtp_input(self) -> None:
        # separate_mtp_input routes the shifted embeddings through
        # mtp_decoder_inputs, which the packed-doc forward never reads.
        with self.assertRaisesRegex(ValueError, r"separate_mtp_input"):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=1,
                    separate_mtp_input=True,
                )
            )

    def test_erndata_cp_accepts_contiguous_allgather(self) -> None:
        # `contiguous_allgather` is a supported layout since the erndata MTP
        # path slices through extract_local_cp_chunks (mode-aware) rather than
        # hard-coding zigzag. It is also *mandatory* for the DSv4 hybrid stack,
        # whose attention layers assert on it under CP.
        cfg = TransformerConfig(
            **self._base_kwargs(
                use_erndata=True,
                num_nextn_predict_layers=1,
                context_parallel_size=2,
                cp_balance_mode="contiguous_allgather",
            )
        )
        self.assertEqual(cfg.cp_balance_mode, "contiguous_allgather")
        self.assertEqual(cfg.context_parallel_size, 2)

    def test_erndata_cp_rejects_contiguous_a2a(self) -> None:
        # Ulysses splits heads inside the attention, so the local-sequence mask
        # contract the MTP path would slice against is not established.
        #
        # Match the erndata-specific wording, not the bare flag name: a plain
        # `cp_balance_mode` regex would also be satisfied by the generic
        # cp_balance_mode validation later in __post_init__, so the test would
        # keep passing even if this guard were deleted.
        with self.assertRaisesRegex(
            ValueError, r"use_erndata=True with MTP \+ context_parallel_size>1"
        ):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=1,
                    context_parallel_size=2,
                    cp_balance_mode="contiguous_a2a",
                )
            )

    def test_erndata_cp_accepts_dualchunk_allgather(self) -> None:
        cfg = TransformerConfig(
            **self._base_kwargs(
                use_erndata=True,
                num_nextn_predict_layers=1,
                context_parallel_size=2,
                cp_balance_mode="dualchunk_allgather",
            )
        )
        self.assertEqual(cfg.cp_balance_mode, "dualchunk_allgather")
        self.assertEqual(cfg.context_parallel_size, 2)

    def test_erndata_cp_rejects_experimental_version(self) -> None:
        # gpt_model_use_experimental_version makes GPTEmbedding build a 2-column
        # attn_mask_startend_row_indices, which the CP mask expansion accepts
        # only under experimental_dataflow -- forbidden with erndata. Without
        # this guard the combination dies as "Invalid attention mask shape"
        # inside DotProductAttention, naming neither flag.
        with self.assertRaisesRegex(
            ValueError, r"gpt_model_use_experimental_version"
        ):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=1,
                    context_parallel_size=2,
                    cp_balance_mode="contiguous_allgather",
                    gpt_model_use_experimental_version=True,
                )
            )

    def test_erndata_experimental_version_ok_without_cp(self) -> None:
        # The 2-column mask is only a problem for the CP expansion path, so
        # CP == 1 must stay legal.
        cfg = TransformerConfig(
            **self._base_kwargs(
                use_erndata=True,
                num_nextn_predict_layers=1,
                gpt_model_use_experimental_version=True,
            )
        )
        self.assertTrue(cfg.gpt_model_use_experimental_version)

    def _dsv4_kwargs(self, **overrides):
        """Smallest config that reaches the dsv4_hybrid validation block.

        ``csa_compress_ratios`` must be num_hidden_layers +
        num_nextn_predict_layers long, and a ratio in [2, 127] with
        ``csa_dense_mode=False`` is what makes gpt_layer_specs build a
        CSAIndexer.
        """
        kwargs = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "use_erndata": True,
            "num_nextn_predict_layers": 1,
            "cp_balance_mode": "contiguous_allgather",
            "experimental_attention_variant": "dsv4_hybrid",
            "csa_compress_ratios": [4, 4, 4],
        }
        kwargs.update(overrides)
        return kwargs

    def test_erndata_cp_rejects_indexer_model(self) -> None:
        # Accepting contiguous_allgather made erndata + MTP + CP + DSv4
        # config-legal for the first time, and that combination is broken
        # downstream: the Indexer loss-mask path all-gathers input_ids and
        # reshapes to [b, cp_size * s_local], but erndata delivers input_ids
        # full-length and global, so the gather over-counts by cp_size. Reject at
        # config time instead of as a shape error inside attention.
        with self.assertRaisesRegex(ValueError, r"Indexer"):
            TransformerConfig(**self._dsv4_kwargs(context_parallel_size=2))

    def test_erndata_indexer_ok_without_cp(self) -> None:
        # CP == 1 never enters the gather branch.
        cfg = TransformerConfig(**self._dsv4_kwargs(context_parallel_size=1))
        self.assertEqual(cfg.context_parallel_size, 1)

    def test_erndata_cp_ok_when_no_indexer_is_built(self) -> None:
        # csa_dense_mode drops the CSAIndexer, and ratio 128 (HCA) never builds
        # one, so neither config reaches the input_ids gather.
        for label, overrides in (
            ("csa_dense_mode", {"csa_dense_mode": True}),
            ("hca_only", {"csa_compress_ratios": [128, 128, 128]}),
        ):
            with self.subTest(label):
                cfg = TransformerConfig(
                    **self._dsv4_kwargs(context_parallel_size=2, **overrides)
                )
                self.assertEqual(cfg.context_parallel_size, 2)

    def test_erndata_accepts_pp_gt_1(self) -> None:
        # PP>1 is supported: cu_seqlens_q is threaded through
        # dist_data_loader.broadcast_data_obj, so the config no longer
        # hard-blocks it. Positive test to lock in the removal of the old block.
        cfg = TransformerConfig(
            **self._base_kwargs(
                use_erndata=True,
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
            )
        )
        self.assertEqual(cfg.pipeline_model_parallel_size, 2)
        self.assertTrue(cfg.use_erndata)

    def test_erndata_cp_rejects_mtp_distillation_loss(self) -> None:
        # LanguageLoss's distillation branch derives lossmask from the
        # already-CP-sliced labels_cur_depth and then scatters it a second time,
        # unlike its non-distillation sibling. Nothing else in __post_init__
        # looked at mtp_distillation_loss, so the combination used to reach a
        # broadcast error at `lossmask * plogp`.
        with self.assertRaisesRegex(ValueError, r"mtp_distillation_loss"):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=1,
                    context_parallel_size=2,
                    cp_balance_mode="contiguous_allgather",
                    mtp_distillation_loss=True,
                )
            )

    def test_erndata_distillation_ok_without_cp(self) -> None:
        # The double scatter only exists on the CP branch.
        cfg = TransformerConfig(
            **self._base_kwargs(
                use_erndata=True,
                num_nextn_predict_layers=1,
                mtp_distillation_loss=True,
            )
        )
        self.assertTrue(cfg.mtp_distillation_loss)

    def test_erndata_cp_requires_a_slicing_path(self) -> None:
        # GPTEmbedding only slices the erndata tensors inside its MTP branch, and
        # the plain-path ContextParallelScatterOp beside it needs
        # experimental_dataflow. With MTP off and the default dataflow nothing
        # shards at all and attention dies on a shape mismatch naming neither
        # flag. Reachable without any MTP-specific setting, since erniebot sets
        # use_erndata implicitly from the YAML's `erndata:` section.
        with self.assertRaisesRegex(ValueError, r"num_nextn_predict_layers"):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=0,
                    context_parallel_size=2,
                )
            )

    def test_erndata_cp_without_mtp_ok_under_experimental_dataflow(
        self,
    ) -> None:
        # The other slicing path: with no MTP layer, GPTEmbedding.forward takes
        # its plain branch, which calls ContextParallelScatterOp on
        # decoder_input (and on the RoPE tables further down) whenever
        # experimental_dataflow is set; LanguageLoss._forward scatters labels
        # under the same flag. The MTP block only forbids experimental_dataflow
        # when num_nextn_predict_layers > 0, so this must stay legal.
        cfg = TransformerConfig(
            **self._base_kwargs(
                use_erndata=True,
                num_nextn_predict_layers=0,
                experimental_dataflow=True,
                context_parallel_size=2,
            )
        )
        self.assertTrue(cfg.use_erndata)
        self.assertTrue(cfg.experimental_dataflow)
        self.assertEqual(cfg.context_parallel_size, 2)

    def test_erndata_cp_without_mtp_ok_under_experimental_dataflow_contiguous(
        self,
    ) -> None:
        # cp_balance_mode is not constrained on that path: it is passed straight
        # through to ContextParallelScatterOp, which implements all three
        # layouts. Only the local-slice MTP path has to agree with the rest of
        # the model, and it is off here.
        for mode in (
            "dualchunk_allgather",
            "contiguous_allgather",
            "contiguous_a2a",
        ):
            with self.subTest(cp_balance_mode=mode):
                cfg = TransformerConfig(
                    **self._base_kwargs(
                        use_erndata=True,
                        num_nextn_predict_layers=0,
                        experimental_dataflow=True,
                        context_parallel_size=2,
                        cp_balance_mode=mode,
                    )
                )
                self.assertEqual(cfg.cp_balance_mode, mode)

    def test_erndata_cp_rejects_weight_only_mtp(self) -> None:
        # WeightOnlyMTPLayer.forward returns its inputs unchanged, so the
        # embedding's slicing branch is gated off and this collapses to the
        # no-MTP case above.
        with self.assertRaisesRegex(ValueError, r"mtp_load_weight_only"):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=1,
                    context_parallel_size=2,
                    cp_balance_mode="contiguous_allgather",
                    mtp_load_weight_only=True,
                )
            )

    def test_erndata_cp_weight_only_mtp_has_no_dataflow_escape(self) -> None:
        # The experimental_dataflow escape does not reach weight-only MTP:
        # num_nextn_predict_layers > 0 puts the config in the MTP block, which
        # rejects experimental_dataflow before the slicing guard is reached. So
        # erndata + CP > 1 + mtp_load_weight_only has no legal spelling, and the
        # rejection has to name experimental_dataflow rather than pass silently.
        with self.assertRaisesRegex(ValueError, r"experimental_dataflow"):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=1,
                    context_parallel_size=2,
                    cp_balance_mode="contiguous_allgather",
                    mtp_load_weight_only=True,
                    experimental_dataflow=True,
                )
            )

    def test_erndata_without_mtp_still_ok_at_cp1(self) -> None:
        # Regression guard for the check above: CP==1 needs no sharding, so the
        # K==0 erndata config must stay legal.
        cfg = TransformerConfig(
            **self._base_kwargs(
                use_erndata=True,
                num_nextn_predict_layers=0,
                context_parallel_size=1,
            )
        )
        self.assertTrue(cfg.use_erndata)

    def test_non_erndata_cp_without_mtp_is_untouched(self) -> None:
        # The new check must not leak into the default dataflow, where
        # ContextParallelScatterOp does the sharding.
        cfg = TransformerConfig(
            **self._base_kwargs(
                num_nextn_predict_layers=0,
                context_parallel_size=2,
            )
        )
        self.assertFalse(cfg.use_erndata)
        self.assertEqual(cfg.context_parallel_size, 2)

    def test_non_erndata_compatible_with_all_flags(self) -> None:
        # Default path must never trip a new guard, even when other MTP flags
        # are enabled (backward-compatible regression test).
        cfg = TransformerConfig(
            **self._base_kwargs(
                num_nextn_predict_layers=1,
                experimental_dataflow=True,
            )
        )
        self.assertFalse(cfg.use_erndata)


if __name__ == "__main__":
    unittest.main()
