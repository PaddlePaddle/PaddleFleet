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

"""Tests for the LanguageLoss cu_seqlens_q stash (per-doc parity path).

Under `use_erndata=True`, `gpt_embedding.forward` stashes
`cu_seqlens_q` onto `LanguageLoss._cu_seqlens_q_stash` so the MTP label-roll
in `LanguageLoss.forward` can call `_roll_tensor_packed_seq` and match the
embedding-side per-doc roll bit-exactly at EOS boundaries.

These tests exercise the roll semantics directly (unit level; no fleet
init) — the class-level stash is a plain module-level slot.
"""

from __future__ import annotations

import numpy as np
import paddle

from paddlefleet.transformer.multi_token_prediction import (
    _roll_tensor_packed_seq,
)


def _plain_roll_with_tail_mask(labels, shifts, ignored_index):
    """Baseline (previous simplified path): paddle.roll left + mask the
    wrapped tail. Used to show the strict per-doc path differs at each
    document's EOS boundary (not just the final tail).
    """
    out = labels
    for _ in range(shifts):
        out = paddle.roll(out, shifts=-1, axis=1)
    out = out.clone()
    out[:, -shifts:] = ignored_index
    return out


def _numpy_per_doc_left_shift(labels_np, cu, shifts):
    """Numpy reference: iterate `shifts` left shifts within each doc,
    zero-out the last position of each doc after each shift. Mirrors what
    the embedding side does token-for-token.
    """
    L = labels_np.shape[1]
    out = labels_np.copy()
    for _ in range(shifts):
        for i in range(len(cu) - 1):
            s, e = int(cu[i]), int(cu[i + 1])
            if e - s <= 0:
                continue
            doc = out[:, s:e]
            rolled = np.roll(doc, shift=-1, axis=1)
            rolled[:, -1] = 0
            out[:, s:e] = rolled
    return out


def test_single_document_no_boundary_effect():
    """cu_seqlens_q = [0, L] (single doc) — packed roll should match the
    plain paddle.roll (aside from the final-tail zero fill vs
    ignored_index).
    """
    paddle.set_device("cpu")
    L = 16
    labels = paddle.arange(L, dtype="int64").reshape([1, L])
    cu = paddle.to_tensor([0, L], dtype="int32")

    rolled_packed, _ = _roll_tensor_packed_seq(
        labels, shifts=-1, dims=1, cu_seqlens_q=cu
    )
    # Plain paddle.roll then zero-fill the tail.
    rolled_plain = paddle.roll(labels, shifts=-1, axis=1).clone()
    rolled_plain[:, -1:] = 0

    assert rolled_packed.numpy().tolist() == rolled_plain.numpy().tolist()


def test_multi_document_matches_numpy_reference():
    """Multi-doc cu_seqlens_q [0, 4, 10, 16] — verify per-doc rolls match
    the token-for-token numpy reference (which the embedding side also
    produces).
    """
    paddle.set_device("cpu")
    labels_np = np.arange(1, 17, dtype=np.int64).reshape([1, 16])
    labels = paddle.to_tensor(labels_np)
    cu = paddle.to_tensor([0, 4, 10, 16], dtype="int32")

    rolled_packed, _ = _roll_tensor_packed_seq(
        labels, shifts=-1, dims=1, cu_seqlens_q=cu
    )
    ref = _numpy_per_doc_left_shift(labels_np, [0, 4, 10, 16], shifts=1)
    assert rolled_packed.numpy().tolist() == ref.tolist()


def test_iterated_rolls_stack_correctly():
    """Rolling K=3 times should still match the numpy reference — this is
    exactly the loop that LanguageLoss now runs for each MTP depth.
    """
    paddle.set_device("cpu")
    labels_np = np.arange(1, 25, dtype=np.int64).reshape([1, 24])
    cu_list = [0, 6, 6, 18, 24]  # includes an empty doc [6:6]
    cu = paddle.to_tensor(cu_list, dtype="int32")

    for depth_plus_one in (1, 2, 3):
        rolled = paddle.to_tensor(labels_np)
        for _ in range(depth_plus_one):
            rolled, _ = _roll_tensor_packed_seq(
                rolled, shifts=-1, dims=1, cu_seqlens_q=cu
            )
        ref = _numpy_per_doc_left_shift(labels_np, cu_list, depth_plus_one)
        assert rolled.numpy().tolist() == ref.tolist(), (
            f"depth+1={depth_plus_one}"
        )


def test_differs_from_plain_roll_at_internal_eos():
    """Sanity: for a multi-doc pack the per-doc roll and the plain
    roll+tail-mask DIFFER at every internal EOS boundary. This is the
    parity gap the stash closes.
    """
    paddle.set_device("cpu")
    labels_np = np.arange(100, 116, dtype=np.int64).reshape([1, 16])
    labels = paddle.to_tensor(labels_np)
    cu = paddle.to_tensor([0, 4, 10, 16], dtype="int32")

    per_doc, _ = _roll_tensor_packed_seq(
        labels, shifts=-1, dims=1, cu_seqlens_q=cu
    )
    plain = _plain_roll_with_tail_mask(labels, shifts=1, ignored_index=-100)

    # Internal EOS positions are index 3 (end of doc 0), 9 (end of doc 1).
    # In per-doc these are zero; in plain they carry the next-doc's first
    # token (or -100 only for the final tail).
    per_doc_l = per_doc.numpy().flatten().tolist()
    plain_l = plain.numpy().flatten().tolist()
    assert per_doc_l[3] == 0
    assert per_doc_l[9] == 0
    assert plain_l[3] != 0  # crosses doc boundary
    assert plain_l[9] != 0


def test_label_roll_fills_ignored_index_at_boundaries():
    """Labels must roll with pad_value=ignored_index (NOT 0). With the
    default pad_value=0 the doc-boundary position becomes token id 0, which
    the loss mask (labels != ignored_index) would treat as a real target and
    train the model to predict token 0 across doc boundaries. pad_value=-100
    makes those positions ignored instead.
    """
    paddle.set_device("cpu")
    ignored_index = -100
    labels_np = np.arange(1, 17, dtype=np.int64).reshape([1, 16])
    labels = paddle.to_tensor(labels_np)
    cu = paddle.to_tensor([0, 4, 10, 16], dtype="int32")

    rolled_lbl, _ = _roll_tensor_packed_seq(
        labels, shifts=-1, dims=1, cu_seqlens_q=cu, pad_value=ignored_index
    )
    rolled_emb, _ = _roll_tensor_packed_seq(
        labels, shifts=-1, dims=1, cu_seqlens_q=cu, pad_value=0
    )
    lbl = rolled_lbl.numpy().flatten().tolist()
    emb = rolled_emb.numpy().flatten().tolist()

    # Boundary positions (last index of each doc): 3, 9, 15.
    for b in (3, 9, 15):
        assert lbl[b] == ignored_index, f"label boundary {b} must be ignored"
        assert emb[b] == 0, f"embedding boundary {b} must be zero-filled"
    # Non-boundary positions are identical between the two fills.
    for i in range(16):
        if i not in (3, 9, 15):
            assert lbl[i] == emb[i]


def test_loss_mask_excludes_every_doc_boundary():
    """The loss mask ``labels != ignored_index`` must drop exactly one
    position per (non-empty) document after a single roll — i.e. the boundary
    tokens do NOT count toward the loss.
    """
    paddle.set_device("cpu")
    ignored_index = -100
    labels_np = np.arange(1, 25, dtype=np.int64).reshape([1, 24])
    labels = paddle.to_tensor(labels_np)
    cu_list = [0, 6, 12, 24]  # 3 non-empty docs
    cu = paddle.to_tensor(cu_list, dtype="int32")

    rolled, _ = _roll_tensor_packed_seq(
        labels, shifts=-1, dims=1, cu_seqlens_q=cu, pad_value=ignored_index
    )
    mask = (rolled != ignored_index).numpy().flatten().tolist()
    # Boundaries 5, 11, 23 must be excluded; everything else kept.
    boundaries = {5, 11, 23}
    for i in range(24):
        assert mask[i] == (i not in boundaries), f"pos {i}"
    assert sum(mask) == 24 - len(boundaries)
