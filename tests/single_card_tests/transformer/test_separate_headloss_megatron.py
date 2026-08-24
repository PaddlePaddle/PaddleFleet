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

"""Single-card coverage for the separate_mtp_headloss megatron label path.

Drives the REAL ``MainLanguageLoss`` / ``MTPLanguageLoss`` / base
``LanguageLoss._megatron_label_for_depth`` via ``__new__`` + MagicMock config
and a stubbed ``_forward`` that records the labels handed to it. Under
use_erndata=True:

  * main labels stay length-L (no ``labels[:, :-K]`` trim);
  * per-MTP-depth labels come from ``_roll_tensor_packed_seq`` with
    ``pad_value=ignored_index``, rolled ``depth+1`` times per packed document
    (boundary positions filled with -100), driven by the class-level
    ``LanguageLoss._cu_seqlens_q_stash``;
  * a missing stash raises RuntimeError instead of silently rolling labels
    across document boundaries.

Also asserts ``GPTMTPLMHead._stash_cu_seqlens_q`` writes the stash so the
loss stage sees cu_seqlens_q under PP>1.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import paddle

from paddlefleet.models.common.language_loss.language_loss import (
    LanguageLoss,
    MainLanguageLoss,
    MTPLanguageLoss,
)

IGNORED = -100


def _cu(cu_list):
    return paddle.to_tensor(cu_list, dtype="int32")


def _ref_per_doc_roll(labels_np, cu_list, depth, ignored=IGNORED):
    """NumPy reference: roll left (depth+1) times per packed doc, boundary=-100.

    ``labels_np`` is [B, L]. Returns the expected labels for MTP depth ``depth``.
    """
    out = labels_np.copy()
    for _ in range(depth + 1):
        rolled = out.copy()
        for i in range(len(cu_list) - 1):
            s, e = cu_list[i], cu_list[i + 1]
            if e - s <= 0:
                continue
            seg = out[:, s:e]
            seg_rolled = np.roll(seg, -1, axis=1)
            seg_rolled[:, -1] = ignored
            rolled[:, s:e] = seg_rolled
        out = rolled
    return out


def _base_loss(K):
    """Bare ``LanguageLoss`` for testing ``_megatron_label_for_depth``."""
    loss = LanguageLoss.__new__(LanguageLoss)
    loss.config = MagicMock()
    loss.config.num_nextn_predict_layers = K
    loss.config.use_erndata = True
    loss.ignored_index = IGNORED
    return loss


def _record_forward():
    """Return (stub, recorded) where stub records the labels it is called with."""
    recorded = []

    def _stub(logits, labels):
        recorded.append(labels)
        return paddle.to_tensor(1.0, dtype="float32")

    return _stub, recorded


class TestMegatronLabelForDepth(unittest.TestCase):
    """Base ``LanguageLoss._megatron_label_for_depth`` (CP=1, TP=1)."""

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_depth_minus_one_is_full_length_main_label(self) -> None:
        # depth < 0 => main labels, unchanged, full length L (no L-K trim).
        loss = _base_loss(K=2)
        L = 16
        cu_list = [0, 4, 10, 16]
        LanguageLoss._cu_seqlens_q_stash = _cu(cu_list)
        labels_np = np.arange(L, dtype="int64").reshape([1, L])
        labels = paddle.to_tensor(labels_np)
        out = loss._megatron_label_for_depth(labels, -1)
        self.assertEqual(list(out.shape), [1, L])
        np.testing.assert_array_equal(out.numpy(), labels_np)

    def test_per_depth_roll_matches_numpy_reference(self) -> None:
        loss = _base_loss(K=2)
        L = 16
        cu_list = [0, 4, 10, 16]
        LanguageLoss._cu_seqlens_q_stash = _cu(cu_list)
        labels_np = np.arange(L, dtype="int64").reshape([1, L])
        labels = paddle.to_tensor(labels_np)
        for depth in range(2):
            out = loss._megatron_label_for_depth(labels, depth)
            self.assertEqual(list(out.shape), [1, L])
            ref = _ref_per_doc_roll(labels_np, cu_list, depth)
            np.testing.assert_array_equal(out.numpy(), ref)

    def test_boundary_positions_are_ignored_index(self) -> None:
        loss = _base_loss(K=2)
        L = 16
        cu_list = [0, 4, 10, 16]
        LanguageLoss._cu_seqlens_q_stash = _cu(cu_list)
        labels = paddle.arange(L, dtype="int64").reshape([1, L])
        # depth 0: last position of each doc (index 3, 9, 15).
        d0 = loss._megatron_label_for_depth(labels, 0).numpy()[0]
        for idx in (3, 9, 15):
            self.assertEqual(d0[idx], IGNORED)
        # depth 1: last two positions of each doc (2,3 / 8,9 / 14,15).
        d1 = loss._megatron_label_for_depth(labels, 1).numpy()[0]
        for idx in (2, 3, 8, 9, 14, 15):
            self.assertEqual(d1[idx], IGNORED)

    def test_missing_stash_raises(self) -> None:
        loss = _base_loss(K=1)
        LanguageLoss._cu_seqlens_q_stash = None
        labels = paddle.arange(16, dtype="int64").reshape([1, 16])
        with self.assertRaises(RuntimeError):
            loss._megatron_label_for_depth(labels, 0)
        # depth < 0 never needs the stash.
        out = loss._megatron_label_for_depth(labels, -1)
        self.assertEqual(list(out.shape), [1, 16])


class TestMTPLanguageLossMegatron(unittest.TestCase):
    """Real ``MTPLanguageLoss.forward`` under use_erndata=True."""

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def _make(self, K):
        loss = MTPLanguageLoss.__new__(MTPLanguageLoss)
        cfg = MagicMock()
        cfg.num_nextn_predict_layers = K
        cfg.mtp_load_weight_only = False
        cfg.use_erndata = True
        cfg.mtp_distillation_loss = False
        loss.config = cfg
        loss.ignored_index = IGNORED
        return loss

    def test_per_depth_labels_are_length_L_and_boundary_masked(self) -> None:
        K, B, L, V = 2, 1, 16, 5
        cu_list = [0, 4, 10, 16]
        loss = self._make(K)
        stub, recorded = _record_forward()
        loss._forward = stub
        LanguageLoss._cu_seqlens_q_stash = _cu(cu_list)

        labels_np = np.arange(L, dtype="int64").reshape([B, L])
        labels = paddle.to_tensor(labels_np)
        mtp_logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K)
        ]
        out = loss.forward({"mtp_logits": mtp_logits, "labels": labels})

        # forward returns dict_args with per-depth mtp_loss, mtp_logits popped.
        self.assertIn("mtp_loss", out)
        self.assertNotIn("mtp_logits", out)
        self.assertEqual(len(out["mtp_loss"]), K)
        # One recorded labels tensor per depth; each length-L and matches ref.
        self.assertEqual(len(recorded), K)
        for depth in range(K):
            got = recorded[depth].numpy()
            self.assertEqual(list(got.shape), [B, L])
            ref = _ref_per_doc_roll(labels_np, cu_list, depth)
            np.testing.assert_array_equal(got, ref)

    def test_missing_stash_raises(self) -> None:
        K, B, L, V = 1, 1, 16, 5
        loss = self._make(K)
        loss._forward, _ = _record_forward()
        LanguageLoss._cu_seqlens_q_stash = None
        labels = paddle.arange(L, dtype="int64").reshape([B, L])
        mtp_logits = [paddle.randn([B, L, V], dtype="float32")]
        with self.assertRaises(RuntimeError):
            loss.forward({"mtp_logits": mtp_logits, "labels": labels})


class TestMainLanguageLossMegatron(unittest.TestCase):
    """Real ``MainLanguageLoss.forward`` under use_erndata=True."""

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def _make(self, K):
        loss = MainLanguageLoss.__new__(MainLanguageLoss)
        cfg = MagicMock()
        cfg.num_nextn_predict_layers = K
        cfg.mtp_load_weight_only = False
        cfg.use_erndata = True
        cfg.mtp_distillation_loss = False
        cfg.train_mtp_only = False
        cfg.add_mtp_loss = True
        cfg.mtp_loss_scaling_factor = 1.0
        loss.config = cfg
        loss.ignored_index = IGNORED
        return loss

    def test_main_label_is_full_length_L(self) -> None:
        K, B, L, V = 2, 1, 16, 5
        cu_list = [0, 4, 10, 16]
        loss = self._make(K)
        stub, recorded = _record_forward()
        loss._forward = stub
        LanguageLoss._cu_seqlens_q_stash = _cu(cu_list)

        labels_np = np.arange(L, dtype="int64").reshape([B, L])
        labels = paddle.to_tensor(labels_np)
        dict_args = {
            "logits": paddle.randn([B, L, V], dtype="float32"),
            "mtp_loss": [
                paddle.to_tensor(1.0, dtype="float32") for _ in range(K)
            ],
        }
        out = loss.forward(dict_args, labels)
        self.assertEqual(out.dtype, paddle.float32)
        # The main label handed to _forward is full-length L, NOT labels[:, :-K],
        # and unchanged (depth == -1 path).
        self.assertEqual(len(recorded), 1)
        got = recorded[0].numpy()
        self.assertEqual(list(got.shape), [B, L])
        np.testing.assert_array_equal(got, labels_np)


class TestGPTMTPLMHeadStash(unittest.TestCase):
    """``GPTMTPLMHead._stash_cu_seqlens_q`` writes the loss-stage stash."""

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_stash_written_from_dict_args(self) -> None:
        from paddlefleet.models.gpt.lm_head import GPTMTPLMHead

        head = GPTMTPLMHead.__new__(GPTMTPLMHead)
        cu = _cu([0, 4, 10, 16])
        head._stash_cu_seqlens_q({"cu_seqlens_q": cu})
        self.assertIsNotNone(LanguageLoss._cu_seqlens_q_stash)
        np.testing.assert_array_equal(
            LanguageLoss._cu_seqlens_q_stash.numpy(), cu.numpy()
        )

    def test_no_cu_seqlens_q_is_noop(self) -> None:
        from paddlefleet.models.gpt.lm_head import GPTMTPLMHead

        head = GPTMTPLMHead.__new__(GPTMTPLMHead)
        head._stash_cu_seqlens_q({"hidden_states": None})
        self.assertIsNone(LanguageLoss._cu_seqlens_q_stash)


if __name__ == "__main__":
    unittest.main()
