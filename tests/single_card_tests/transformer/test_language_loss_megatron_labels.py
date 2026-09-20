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
and replacing ``extract_local_cp_chunks`` / the CP comm ops with
identities (see TestLanguageLossMegatronCP).

``_megatron_label_for_depth`` -- the separate Main/MTP head-loss entry point,
which reaches its own CP slice without going through ``forward`` -- is covered
by TestMegatronLabelForDepthCP, which runs the REAL extract helper so the
slice is checked by value rather than by recorded kwarg.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
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


def _make_loss(
    K, *, distill, use_erndata=True, cp_balance_mode="dualchunk_allgather"
):
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
    cfg.cp_balance_mode = cp_balance_mode
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
    - ``extract_local_cp_chunks`` (local import) -> identity, recording the
      kwargs it was called with;
    - CP scatter/gather PyLayers -> identity;
    - ``dist.all_reduce`` -> no-op and ``fleet`` -> MagicMock (the
      distillation branch all-reduces the per-depth loss).

    Yields the recorded ``extract_local_cp_chunks`` calls.
    """

    calls = []

    def identity(t, *a, **k):
        return t

    def recording_identity(t, *a, **k):
        calls.append((a, k))
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
            mock.patch.object(
                mtp, "extract_local_cp_chunks", recording_identity
            )
        )
        stack.enter_context(
            mock.patch.object(ll.ContextParallelScatterOp, "apply", identity)
        )
        stack.enter_context(
            mock.patch.object(ll.ContextParallelGatherOp, "apply", identity)
        )

        # The distillation branch of LanguageLoss.forward swaps the gather for
        # MTPDistillationLossShift under contiguous_allgather. That PyLayer
        # needs a real hybrid communicate group, so stand in for it with a
        # shape-faithful stub: it consumes [B, S, H] and returns
        # [B, S + K - 1, H] (the local slice plus the boundary window its P2P
        # exchange fetches), which is what the per-depth
        # ``target_p_self_op_dist[:, depth : depth + out_logp.shape[1]]`` slice
        # downstream assumes.
        def shift_stub(tensor, num_nextn_predict_layers, *a, **k):
            b, _s, h = tensor.shape
            tail = paddle.zeros(
                [b, num_nextn_predict_layers, h], dtype=tensor.dtype
            )
            return paddle.concat([tensor[:, 1:], tail], axis=1)

        stack.enter_context(
            mock.patch.object(ll.MTPDistillationLossShift, "apply", shift_stub)
        )
        stack.enter_context(
            mock.patch.object(ll.dist, "all_reduce", lambda *a, **k: None)
        )
        stack.enter_context(mock.patch.object(ll, "fleet", MagicMock()))
        yield calls


class TestLanguageLossMegatronCP(unittest.TestCase):
    """Covers the CP>1 sublines (515,518,522,530,603,771) via monkeypatch.

    ``extract_local_cp_chunks`` is mocked to identity, so all tensors
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


class TestLanguageLossCpBalanceMode(unittest.TestCase):
    """Every CP slice here must use ``config.cp_balance_mode``.

    The identity mock above keeps shapes self-consistent whatever layout is
    requested, so the two tests before this one pass even if the mode is
    hard-coded -- which is exactly the defect this change fixes. Assert on the
    recorded kwarg instead, for both layouts and both loss branches.
    """

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def _modes_used(self, *, distill, cp_balance_mode):
        K, B, L, V = 2, 1, 8, 5
        loss = _make_loss(K, distill=distill, cp_balance_mode=cp_balance_mode)
        LanguageLoss._cu_seqlens_q_stash = _make_cu([3, 5])
        logits = [
            paddle.randn([B, L, V], dtype="float32") for _ in range(K + 1)
        ]
        labels = paddle.arange(B * L, dtype="int64").reshape([B, L])
        with _fake_cp(cp_size=2) as calls:
            loss.forward(logits, labels)
        self.assertTrue(calls, "no CP slice happened, so nothing was checked")
        return [kwargs.get("mode") for _args, kwargs in calls]

    def test_non_distill_forwards_configured_mode(self) -> None:
        for mode in ("dualchunk_allgather", "contiguous_allgather"):
            with self.subTest(mode=mode):
                used = self._modes_used(distill=False, cp_balance_mode=mode)
                self.assertEqual(set(used), {mode})

    def test_distill_forwards_configured_mode(self) -> None:
        for mode in ("dualchunk_allgather", "contiguous_allgather"):
            with self.subTest(mode=mode):
                used = self._modes_used(distill=True, cp_balance_mode=mode)
                self.assertEqual(set(used), {mode})


@contextlib.contextmanager
def _cp_ranks(cp_size, cp_rank):
    """Fake a CP group without touching ``extract_local_cp_chunks``.

    Unlike ``_fake_cp`` this leaves the real extract helper installed, so the
    slice actually happens and can be asserted on by value. Only the two
    lookups ``_megatron_label_for_depth`` performs are patched: the module-level
    ``get_context_parallel_world_size`` and the locally imported
    ``get_context_parallel_rank``.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                ll, "get_context_parallel_world_size", lambda: cp_size
            )
        )
        stack.enter_context(
            mock.patch.object(ps, "get_context_parallel_rank", lambda: cp_rank)
        )
        yield


def _local_slice_ref(full_np, cp_rank, cp_size, mode):
    """NumPy reference for the two supported CP layouts on the seq axis.

    ``dualchunk_allgather`` mirrors ``scatter_balance``: rank r owns
    ``[interval*r, interval*(r+1))`` plus the mirrored tail chunk.
    ``contiguous_allgather`` mirrors ``scatter_contiguous``: one chunk per rank.
    """
    seq_len = full_np.shape[1]
    if mode == "dualchunk_allgather":
        interval = seq_len // cp_size // 2
        head = full_np[:, interval * cp_rank : interval * (cp_rank + 1)]
        tail = full_np[
            :, seq_len - interval * (cp_rank + 1) : seq_len - interval * cp_rank
        ]
        return np.concatenate([head, tail], axis=1)
    chunk = seq_len // cp_size
    return full_np[:, chunk * cp_rank : chunk * (cp_rank + 1)]


class TestMegatronLabelForDepthCP(unittest.TestCase):
    """``_megatron_label_for_depth`` slices with the REAL extract helper.

    The separate Main/MTP head-loss path (``GPTMainLMHead`` / ``GPTMTPLMHead``,
    language_loss.py:1026 and 1128) reaches the CP slice through this method
    rather than through ``forward``, so the identity mock used above never
    exercises it. Here the real ``extract_local_cp_chunks`` runs and the
    assertion is on values: the CP>1 label must equal the numpy slice of the
    CP=1 label for the configured ``cp_balance_mode``, on every rank and at
    every depth. A hard-coded layout would fail one of the two modes, and the
    union over ranks would not reconstruct the full sequence.
    """

    L = 16
    CU = (4, 6, 6)

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def _labels(self):
        labels_np = np.arange(self.L, dtype="int64").reshape([1, self.L])
        return labels_np, paddle.to_tensor(labels_np)

    def test_slice_matches_cp1_label_for_both_modes(self) -> None:
        cp_size = 2
        for mode in ("dualchunk_allgather", "contiguous_allgather"):
            for depth in (-1, 0, 1):
                loss = _make_loss(2, distill=False, cp_balance_mode=mode)
                LanguageLoss._cu_seqlens_q_stash = _make_cu(list(self.CU))
                _labels_np, labels = self._labels()
                # CP=1 gives the full-length label this depth should shard.
                full = loss._megatron_label_for_depth(labels, depth).numpy()
                for cp_rank in range(cp_size):
                    with self.subTest(mode=mode, depth=depth, rank=cp_rank):
                        with _cp_ranks(cp_size, cp_rank):
                            got = loss._megatron_label_for_depth(
                                labels, depth
                            ).numpy()
                        self.assertEqual(
                            list(got.shape), [1, self.L // cp_size]
                        )
                        np.testing.assert_array_equal(
                            got,
                            _local_slice_ref(full, cp_rank, cp_size, mode),
                        )

    def test_ranks_together_cover_the_full_label(self) -> None:
        # Whatever the layout, the ranks partition the sequence: no token is
        # dropped or counted twice, so every label value appears exactly once.
        cp_size = 2
        for mode in ("dualchunk_allgather", "contiguous_allgather"):
            with self.subTest(mode=mode):
                loss = _make_loss(2, distill=False, cp_balance_mode=mode)
                LanguageLoss._cu_seqlens_q_stash = _make_cu(list(self.CU))
                _labels_np, labels = self._labels()
                seen = []
                for cp_rank in range(cp_size):
                    with _cp_ranks(cp_size, cp_rank):
                        seen.append(
                            loss._megatron_label_for_depth(labels, -1).numpy()
                        )
                union = np.sort(np.concatenate(seen, axis=1), axis=1)
                np.testing.assert_array_equal(
                    union, np.arange(self.L, dtype="int64").reshape([1, self.L])
                )

    def test_unsupported_mode_is_rejected(self) -> None:
        # contiguous_a2a shards contiguously but has a different mask contract
        # and has never run on this path, so the helper refuses rather than
        # silently returning a slice that would train against wrong labels.
        loss = _make_loss(2, distill=False, cp_balance_mode="contiguous_a2a")
        LanguageLoss._cu_seqlens_q_stash = _make_cu(list(self.CU))
        _labels_np, labels = self._labels()
        with (
            _cp_ranks(2, 0),
            self.assertRaisesRegex(ValueError, r"cp_balance_mode"),
        ):
            loss._megatron_label_for_depth(labels, 0)

    def test_cp1_returns_full_length_unchanged(self) -> None:
        # The CP block is skipped entirely at cp_size == 1, so depth -1 hands
        # back the caller's own tensor untouched.
        loss = _make_loss(2, distill=False)
        LanguageLoss._cu_seqlens_q_stash = _make_cu(list(self.CU))
        labels_np, labels = self._labels()
        with _cp_ranks(1, 0):
            out = loss._megatron_label_for_depth(labels, -1)
        self.assertEqual(list(out.shape), [1, self.L])
        np.testing.assert_array_equal(out.numpy(), labels_np)


if __name__ == "__main__":
    unittest.main()
