# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TransformerConfig.use_erndata validation branches.

``use_erndata`` selects the MTP data-flow contract: True means the erndata
(Energon) pipeline emits length-L tensors + cu_seqlens_q, False means the
historical ernie5 L+K layout. Guards checked here:
  1. Default is False.
  2. use_erndata + MTP requires num_nextn_predict_layers > 0 (the
     `mtp_num_layers` alias is deliberately NOT accepted).
  3. use_erndata + MTP is incompatible with enable_mtp_magic_send.
  4. use_erndata + MTP is incompatible with experimental_dataflow.
  5. use_erndata without MTP (K == 0) trips none of the guards — the packed-doc
     forward is only reachable through the MTP layer.
  6. use_erndata=False never trips any of the guards regardless of other MTP
     flags.
"""

from __future__ import annotations

import unittest

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

    def test_erndata_rejects_mtp_num_layers_alias_only(self) -> None:
        # `mtp_num_layers` is honored by MTP layer *construction*
        # (_get_effective_mtp_layers) but not by the erndata data path, which
        # reads num_nextn_predict_layers everywhere. Configuring K only through
        # the alias would build MTP layers that never receive shifted
        # embeddings, so it must be rejected rather than silently accepted.
        with self.assertRaisesRegex(ValueError, r"num_nextn_predict_layers"):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=0,
                    mtp_num_layers=2,
                )
            )

    def test_erndata_incompat_with_magic_send(self) -> None:
        # enable_mtp_magic_send also requires PP>1 (checked earlier in
        # __post_init__), so we build a config that would pass that check.
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

    def test_erndata_cp_requires_dualchunk_allgather(self) -> None:
        # `contiguous_allgather` is not equivalent to MCore's zigzag layout.
        with self.assertRaisesRegex(
            ValueError, r"cp_balance_mode='dualchunk_allgather'"
        ):
            TransformerConfig(
                **self._base_kwargs(
                    use_erndata=True,
                    num_nextn_predict_layers=1,
                    context_parallel_size=2,
                    cp_balance_mode="contiguous_allgather",
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
