# Copyright (c) 2026 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""PP>1 support for cu_seqlens_q under use_erndata=True.

Two invariants pinned here:

  1. `cu_seqlens_q` (int32 [num_docs+1]) round-trips through
     `broadcast_data_obj` unchanged when mixed into a tuple alongside
     int64 / bfloat16 / None entries — same shape / dtype / values on
     receiver side. This is the primitive the dataloader relies on.

  2. `LanguageLoss._cu_seqlens_q_stash` is a plain class-level slot
     that can be written and read across module-level references
     without falling out of scope. This is the mechanism by which the
     last PP stage (which never runs `gpt_embedding.forward`) sees
     `cu_seqlens_q` — the dataloader writes it on every rank right
     after the three `broadcast_data_obj` calls.

See ernie5/src/datasets/dist_data_loader.py.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.models.common.language_loss.language_loss import (
    LanguageLoss,
)
from paddlefleet.pipeline_parallel.pp_utils.pp_comm_utils import (
    broadcast_data_obj,
)


class TestCuSeqlensQBroadcast(unittest.TestCase):
    """cu_seqlens_q must survive broadcast_data_obj alongside other dtypes."""

    def test_cu_seqlens_q_roundtrips_alongside_mixed_dtypes(self) -> None:
        # Emulate a slice of the real dataloader tuple: int64 ids/labels,
        # bfloat16 image scale, None for absent optional, int32 cu_seqlens_q.
        cu = paddle.to_tensor([0, 4, 10, 16], dtype="int32")
        input_ids = paddle.arange(16, dtype="int64").reshape([1, 16])
        labels = paddle.arange(16, dtype="int64").reshape([1, 16])
        image_scale = paddle.zeros([1], dtype="bfloat16")

        # Sender side — broadcast_data_obj concats same-dtype tensors, so
        # this exercises the int32 group (cu_seqlens_q alone in its dtype
        # bucket among these fields).
        with (
            patch("paddle.distributed.get_rank", return_value=0),
            patch("paddle.distributed.broadcast_object_list"),
            patch("paddle.distributed.broadcast"),
        ):
            out = broadcast_data_obj(
                [input_ids, labels, image_scale, None, cu],
                src_rank=0,
                group=MagicMock(),
            )
        # Sender path returns the same values (no receive branch invoked).
        self.assertIsNone(out[3])
        self.assertEqual(out[4].dtype, paddle.int32)
        self.assertEqual(out[4].shape, [4])
        self.assertEqual(out[4].numpy().tolist(), [0, 4, 10, 16])


class TestLanguageLossStashAcrossModules(unittest.TestCase):
    """The dataloader writes stash on every rank; loss reads it later."""

    def setUp(self) -> None:
        # Ensure we don't leak state from other tests.
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_stash_survives_reimport(self) -> None:
        cu = paddle.to_tensor([0, 4, 10, 16], dtype="int32")
        # First writer (simulating dist_data_loader.py which imports lazily).
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss as _LL_writer,
        )

        _LL_writer._cu_seqlens_q_stash = cu

        # Second reader path (simulating LanguageLoss.forward reading it).
        from paddlefleet.models.common.language_loss.language_loss import (
            LanguageLoss as _LL_reader,
        )

        self.assertIs(_LL_reader._cu_seqlens_q_stash, cu)
        self.assertEqual(_LL_reader._cu_seqlens_q_stash.dtype, paddle.int32)
        self.assertEqual(
            _LL_reader._cu_seqlens_q_stash.numpy().tolist(),
            [0, 4, 10, 16],
        )

    def test_stash_none_is_the_default(self) -> None:
        # Default class-level value must be None so that loss falls back
        # to the plain paddle.roll path for the ernie5 flow.
        self.assertIsNone(LanguageLoss._cu_seqlens_q_stash)


if __name__ == "__main__":
    unittest.main()
