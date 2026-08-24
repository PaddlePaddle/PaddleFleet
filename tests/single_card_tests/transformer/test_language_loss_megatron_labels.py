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

"""Single-card coverage for LanguageLoss.forward under use_erndata=True.

Drives the REAL ``LanguageLoss.forward`` via ``__new__`` + MagicMock config
and stubbed ``_forward`` / ``loss_func`` so the megatron label-shift paths
are exercised on a single GPU (CP=1, TP=1):

- non-distillation megatron branch: full-length ``lm_labels = labels_ori``,
  per-depth ``_roll_tensor_packed_seq`` driven by the class-level
  ``_cu_seqlens_q_stash`` (language_loss.py:527-528, 536, 566-574, 600, 609).
- distillation megatron branch: mirror roll path (lines 742-750, 770, 777).
- missing-stash guards raise RuntimeError in both branches (lines 592, 762).

The CP>1 sublines (515/518/522/530/603/771) are covered single-card by
monkeypatching ``get_context_parallel_world_size`` -> 2, faking the CP rank,
and replacing ``extract_local_zigzag_chunks`` / the CP comm ops with
identities (see TestLanguageLossMegatronCP).
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock
from unittest.mock import MagicMock

import paddle

import paddlefleet.models.common.language_loss.language_loss as ll
import paddlefleet.parallel_state as ps
import paddlefleet.transformer.multi_token_prediction as mtp
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss


def _make_cu(seq_lens):
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + n)
    return paddle.to_tensor(cu, dtype="int32")


def _make_loss(K, *, distill, use_erndata=True):
    loss = LanguageLoss.__new__(LanguageLoss)
    cfg = MagicMock()
    cfg.num_nextn_predict_layers = K
    cfg.mtp_load_weight_only = False
    cfg.use_erndata = use_erndata
    cfg.mtp_distillation_loss = distill
    cfg.train_mtp_only = False
    cfg.gpt_model_use_experimental_version = True
    cfg.sequence_parallel = False
    cfg.fused_linear_ce_loss_chunk = 0
    cfg.add_mtp_loss = True
    cfg.mtp_loss_scaling_factor = 1.0
    cfg.experimental_dataflow = False
    cfg.cp_balance_mode = "zigzag"
    cfg.recompute_modules = None
    loss.config = cfg
    loss.ignored_index = -100

    def _stub_forward(logits, labels):
        return paddle.to_tensor(1.0, dtype="float32")

    def _stub_loss_func(logits, labels):
        # Per-token loss matrix shaped like labels [B, L].
        return paddle.ones(labels.shape, dtype="float32")

    loss._forward = _stub_forward
    loss.loss_func = _stub_loss_func
    return loss


class TestLanguageLossMegatronNonDistill(unittest.TestCase):
    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_non_distill_with_stash(self) -> None:
        K, B, L, V = 2, 1, 8, 5
        loss = _make_loss(K, distill=False)
        LanguageLoss._cu_seqlens_q_stash = _make_cu([3, 5])
        logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K + 1)
        ]
        labels = paddle.arange(B * L, dtype="int64").reshape([B, L])
        out = loss.forward(logits, labels)
        self.assertEqual(out.dtype, paddle.float32)
        # scalar loss
        self.assertEqual(list(out.shape), [] if out.ndim == 0 else [1])

    def test_non_distill_missing_stash_raises(self) -> None:
        K, B, L, V = 1, 1, 8, 5
        loss = _make_loss(K, distill=False)
        LanguageLoss._cu_seqlens_q_stash = None
        logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K + 1)
        ]
        labels = paddle.arange(B * L, dtype="int64").reshape([B, L])
        with self.assertRaises(RuntimeError):
            loss.forward(logits, labels)


class TestLanguageLossMegatronDistill(unittest.TestCase):
    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_distill_with_stash(self) -> None:
        K, B, L, V = 2, 1, 8, 5
        loss = _make_loss(K, distill=True)
        LanguageLoss._cu_seqlens_q_stash = _make_cu([4, 4])
        logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K + 1)
        ]
        labels = paddle.arange(B * L, dtype="int64").reshape([B, L])
        out = loss.forward(logits, labels)
        self.assertEqual(out.dtype, paddle.float32)

    def test_distill_missing_stash_raises(self) -> None:
        K, B, L, V = 1, 1, 8, 5
        loss = _make_loss(K, distill=True)
        LanguageLoss._cu_seqlens_q_stash = None
        logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K + 1)
        ]
        labels = paddle.arange(B * L, dtype="int64").reshape([B, L])
        with self.assertRaises(RuntimeError):
            loss.forward(logits, labels)


class TestLanguageLossErnie5Slice(unittest.TestCase):
    """ernie5 (non-megatron) MTP label path: lm_labels = labels[:, :-K] and
    per-depth labels_cur_depth = labels_ori[:, (depth+1):(depth+1+seq)]
    (language_loss.py:538-539, 611). Single-card, CP=1, TP=1.
    """

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_ernie5_non_distill_slice(self) -> None:
        K, B, L, V = 2, 1, 10, 5
        loss = _make_loss(K, distill=False, use_erndata=False)
        logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K + 1)
        ]
        labels = paddle.arange(B * L, dtype="int64").reshape([B, L])
        out = loss.forward(logits, labels)
        self.assertEqual(out.dtype, paddle.float32)


@contextlib.contextmanager
def _fake_cp(cp_size=2):
    """Monkeypatch the CP machinery so the ``_cp_size_for_extract > 1``
    branches run single-card:

    - module-level ``get_context_parallel_world_size`` -> cp_size;
    - source-module ``get_context_parallel_rank`` (local import) -> 0;
    - ``extract_local_zigzag_chunks`` (local import) -> identity;
    - CP scatter/gather PyLayers -> identity;
    - ``dist.all_reduce`` -> no-op and ``fleet`` -> MagicMock (the
      distillation branch all-reduces the per-depth loss).
    """

    def identity(t, *a, **k):
        return t

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                ll, "get_context_parallel_world_size", lambda: cp_size
            )
        )
        stack.enter_context(
            mock.patch.object(ps, "get_context_parallel_rank", lambda: 0)
        )
        stack.enter_context(
            mock.patch.object(mtp, "extract_local_zigzag_chunks", identity)
        )
        stack.enter_context(
            mock.patch.object(ll.ContextParallelScatterOp, "apply", identity)
        )
        stack.enter_context(
            mock.patch.object(ll.ContextParallelGatherOp, "apply", identity)
        )
        stack.enter_context(
            mock.patch.object(ll.dist, "all_reduce", lambda *a, **k: None)
        )
        stack.enter_context(mock.patch.object(ll, "fleet", MagicMock()))
        yield


class TestLanguageLossMegatronCP(unittest.TestCase):
    """Covers the CP>1 sublines (515,518,522,530,603,771) via monkeypatch.

    ``extract_local_zigzag_chunks`` is mocked to identity, so all tensors
    keep their full length and shapes stay self-consistent.
    """

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_non_distill_cp_branch(self) -> None:
        # Covers 515, 518, 522, 530 (lm_labels extract) and 603 (per-depth).
        K, B, L, V = 2, 1, 8, 5
        loss = _make_loss(K, distill=False)
        LanguageLoss._cu_seqlens_q_stash = _make_cu([3, 5])
        logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K + 1)
        ]
        labels = paddle.arange(B * L, dtype="int64").reshape([B, L])
        with _fake_cp(cp_size=2):
            out = loss.forward(logits, labels)
        self.assertEqual(out.dtype, paddle.float32)

    def test_distill_cp_branch(self) -> None:
        # Covers 771 (per-depth extract in the distillation branch).
        K, B, L, V = 2, 1, 8, 5
        loss = _make_loss(K, distill=True)
        LanguageLoss._cu_seqlens_q_stash = _make_cu([4, 4])
        logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K + 1)
        ]
        labels = paddle.arange(B * L, dtype="int64").reshape([B, L])
        with _fake_cp(cp_size=2):
            out = loss.forward(logits, labels)
        self.assertEqual(out.dtype, paddle.float32)


if __name__ == "__main__":
    unittest.main()
