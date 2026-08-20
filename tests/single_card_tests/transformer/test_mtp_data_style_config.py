# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TransformerConfig.mtp_data_style validation branches.

Guards the new field's __post_init__ checks:
  1. Invalid string is rejected.
  2. "megatron" requires num_nextn_predict_layers > 0 or mtp_num_layers > 0.
  3. "megatron" is incompatible with enable_mtp_magic_send.
  4. "megatron" is incompatible with experimental_dataflow.
  5. "ernie5" (default) never trips any of the new guards regardless of
     other MTP flags.
"""

from __future__ import annotations

import unittest

from paddlefleet.transformer.transformer_config import TransformerConfig


class TestMtpDataStyleValidation(unittest.TestCase):
    """The new mtp_data_style field must be validated in __post_init__."""

    def _base_kwargs(self, **overrides):
        """Minimal kwargs to build a TransformerConfig without noise from
        other fields. All fields have defaults; we only override MTP-related
        ones needed for a given test case."""
        return dict(overrides)

    def test_default_style_is_ernie5(self) -> None:
        cfg = TransformerConfig(**self._base_kwargs())
        self.assertEqual(cfg.mtp_data_style, "ernie5")

    def test_invalid_style_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"mtp_data_style="):
            TransformerConfig(**self._base_kwargs(mtp_data_style="invalid"))

    def test_megatron_requires_positive_mtp_k(self) -> None:
        # Neither num_nextn_predict_layers nor mtp_num_layers > 0 → fail
        with self.assertRaisesRegex(ValueError, r"mtp_data_style='megatron'"):
            TransformerConfig(
                **self._base_kwargs(
                    mtp_data_style="megatron",
                    num_nextn_predict_layers=0,
                    mtp_num_layers=0,
                )
            )

    def test_megatron_accepts_num_nextn(self) -> None:
        cfg = TransformerConfig(
            **self._base_kwargs(
                mtp_data_style="megatron",
                num_nextn_predict_layers=1,
            )
        )
        self.assertEqual(cfg.mtp_data_style, "megatron")

    def test_megatron_accepts_mtp_num_layers(self) -> None:
        cfg = TransformerConfig(
            **self._base_kwargs(
                mtp_data_style="megatron",
                num_nextn_predict_layers=0,
                mtp_num_layers=2,
            )
        )
        self.assertEqual(cfg.mtp_data_style, "megatron")

    def test_megatron_incompat_with_magic_send(self) -> None:
        # enable_mtp_magic_send also requires PP>1 (checked earlier in
        # __post_init__), so we build a config that would pass that check.
        with self.assertRaisesRegex(ValueError, r"enable_mtp_magic_send"):
            TransformerConfig(
                **self._base_kwargs(
                    mtp_data_style="megatron",
                    num_nextn_predict_layers=1,
                    enable_mtp_magic_send=True,
                    pipeline_model_parallel_size=2,
                )
            )

    def test_megatron_incompat_with_experimental_dataflow(self) -> None:
        with self.assertRaisesRegex(ValueError, r"experimental_dataflow"):
            TransformerConfig(
                **self._base_kwargs(
                    mtp_data_style="megatron",
                    num_nextn_predict_layers=1,
                    experimental_dataflow=True,
                )
            )

    def test_megatron_cp_requires_dualchunk_allgather(self) -> None:
        # `contiguous_allgather` is not equivalent to MCore's zigzag layout.
        with self.assertRaisesRegex(
            ValueError, r"cp_balance_mode='dualchunk_allgather'"
        ):
            TransformerConfig(
                **self._base_kwargs(
                    mtp_data_style="megatron",
                    num_nextn_predict_layers=1,
                    context_parallel_size=2,
                    cp_balance_mode="contiguous_allgather",
                )
            )

    def test_megatron_cp_accepts_dualchunk_allgather(self) -> None:
        cfg = TransformerConfig(
            **self._base_kwargs(
                mtp_data_style="megatron",
                num_nextn_predict_layers=1,
                context_parallel_size=2,
                cp_balance_mode="dualchunk_allgather",
            )
        )
        self.assertEqual(cfg.cp_balance_mode, "dualchunk_allgather")
        self.assertEqual(cfg.context_parallel_size, 2)

    def test_megatron_accepts_pp_gt_1(self) -> None:
        # PP>1 is supported: cu_seqlens_q is threaded through
        # dist_data_loader.broadcast_data_obj, so the config no longer hard-blocks
        # it. Positive test to lock in the removal of the old hard block.
        cfg = TransformerConfig(
            **self._base_kwargs(
                mtp_data_style="megatron",
                num_nextn_predict_layers=1,
                pipeline_model_parallel_size=2,
            )
        )
        self.assertEqual(cfg.pipeline_model_parallel_size, 2)
        self.assertEqual(cfg.mtp_data_style, "megatron")

    def test_ernie5_compatible_with_all_flags(self) -> None:
        # Default path must never trip a new guard, even when other MTP flags
        # are enabled (backward-compatible regression test).
        cfg = TransformerConfig(
            **self._base_kwargs(
                num_nextn_predict_layers=1,
                experimental_dataflow=True,
            )
        )
        self.assertEqual(cfg.mtp_data_style, "ernie5")


if __name__ == "__main__":
    unittest.main()
