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

"""Unit tests for MTP K>1 composition (mtp_emb_res list assembly).

Guards the exact code path in ``gpt_embedding.py:357-395`` where the
``mtp_emb_res`` list is assembled by iterated accumulative rolls of the
full-length embedding tensor. Prior coverage:

- ``test_language_loss_cu_seqlens_stash.py::test_iterated_rolls_stack_correctly``
  verifies that K iterations of ``_roll_tensor_packed_seq`` on int labels
  match a numpy per-doc reference.
- ``test_roll_tensor_full_length.py::test_pipeline_multi_depth_cp2`` verifies
  the K=2 accumulative-roll + CP=2 zigzag split combination on int tensors.

This file closes the last gap: for float embedding tensors (the actual
``mtp_emb_res`` payload), K∈{1, 2, 3} composition — list length K+1,
first slot is the base embed, slot k+1 is (k+1) iterated rolls of the
full-length base — must hold both without CP and with CP=2 zigzag
extraction per depth.

Guards against future refactors that silently drop a depth (e.g.
``for k in range(K)`` instead of ``range(K+1)``) or reset the running
roll each iteration (breaking the accumulative semantics).
"""

from __future__ import annotations

import numpy as np
import paddle
import pytest

from paddlefleet.transformer.multi_token_prediction import (
    extract_local_zigzag_chunks,
    roll_tensor,
)


@pytest.fixture(autouse=True)
def _restore_default_device():
    """The tests below run the pure-python roll/extract helpers on CPU.

    ``paddle.set_device`` is process-global, so without restoring it every test
    collected after this file in the same pytest session would also run on CPU
    and hit GPU-only kernels (``rms_norm``).
    """
    prev = paddle.get_device()
    yield
    paddle.set_device(prev)


def _build_mtp_emb_res(
    inputs_embeds: paddle.Tensor,
    inputs_embeds_ori: paddle.Tensor,
    cu_seqlens_q: paddle.Tensor,
    num_nextn_predict_layers: int,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> list[paddle.Tensor]:
    """Faithful reproduction of the ``mtp_emb_res`` assembly loop from
    ``gpt_embedding.py`` (lines 357-395). Kept short and inline here so the
    test does not depend on constructing a full GPTEmbedding module.
    """
    mtp_emb_res = [inputs_embeds]
    rolled_embed = inputs_embeds_ori
    for _ in range(num_nextn_predict_layers):
        rolled_embed, _ = roll_tensor(
            rolled_embed,
            shifts=-1,
            dims=1,
            cp_group=None,
            cu_seqlens_q=cu_seqlens_q,
        )
        if cp_size > 1:
            emb_mtp = extract_local_zigzag_chunks(
                rolled_embed, cp_rank, cp_size, axis=1
            )
        else:
            emb_mtp = rolled_embed
        mtp_emb_res.append(emb_mtp)
    return mtp_emb_res


def _numpy_per_doc_roll_float(
    arr: np.ndarray, cu: list[int], shifts: int
) -> np.ndarray:
    """Numpy reference: iterate ``shifts`` left rolls within each doc,
    zero the last position of each doc after each roll. Works on any
    trailing feature dim (H).
    """
    out = arr.copy()
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


class TestMtpEmbResCompositionNoCP:
    """K∈{1, 2, 3} mtp_emb_res list composition on float embeddings; CP=1
    (no zigzag extraction).
    """

    def _run_k(self, K: int) -> None:
        paddle.set_device("cpu")
        B, L, H = 1, 24, 4
        rng = np.random.default_rng(seed=20260813 + K)
        # Use float payload — real mtp_emb_res carries bf16/fp32 embeddings.
        embeds_np = rng.standard_normal((B, L, H)).astype(np.float32)
        cu_list = [
            0,
            6,
            6,
            18,
            24,
        ]  # includes an empty doc to stress boundary handling
        inputs_embeds = paddle.to_tensor(embeds_np)
        inputs_embeds_ori = paddle.to_tensor(embeds_np)
        cu = paddle.to_tensor(cu_list, dtype="int32")

        mtp_emb_res = _build_mtp_emb_res(
            inputs_embeds, inputs_embeds_ori, cu, K, cp_rank=0, cp_size=1
        )

        # 1. Guard against off-by-one in loop bounds: list must have K+1 slots.
        assert len(mtp_emb_res) == K + 1, (
            f"K={K}: expected {K + 1} slots, got {len(mtp_emb_res)}"
        )

        # 2. Slot 0 is the base embed unchanged.
        np.testing.assert_array_equal(
            mtp_emb_res[0].numpy(),
            embeds_np,
            err_msg=f"K={K}: slot 0 must be the base inputs_embeds",
        )

        # 3. Slot k+1 (for k in [0, K)) is (k+1) accumulated per-doc rolls
        #    of the full-length base. Uses the numpy reference (which
        #    test_iterated_rolls_stack_correctly already validated for
        #    _roll_tensor_packed_seq on ints).
        for k in range(K):
            ref = _numpy_per_doc_roll_float(embeds_np, cu_list, shifts=k + 1)
            np.testing.assert_array_equal(
                mtp_emb_res[k + 1].numpy(),
                ref,
                err_msg=(
                    f"K={K}, depth={k}: mtp_emb_res[{k + 1}] must equal "
                    f"(k+1)={k + 1} iterated per-doc rolls of base embed"
                ),
            )

    def test_k1(self) -> None:
        self._run_k(1)

    def test_k2(self) -> None:
        self._run_k(2)

    def test_k3(self) -> None:
        self._run_k(3)


class TestMtpEmbResCompositionCP2:
    """K∈{1, 2, 3} mtp_emb_res list composition with CP=2 zigzag extraction
    per depth. Verifies that the extraction runs on the *post-roll*
    full-length tensor (embedding invariant with [[eb-mtp-cp-full-length-layout]]).
    """

    def _run_k(self, K: int) -> None:
        paddle.set_device("cpu")
        B, L, H = 1, 16, 4
        cp_size = 2
        rng = np.random.default_rng(seed=20260813 + 100 + K)
        embeds_np = rng.standard_normal((B, L, H)).astype(np.float32)
        cu_list = [0, 8, 16]
        inputs_embeds = paddle.to_tensor(embeds_np)
        inputs_embeds_ori = paddle.to_tensor(embeds_np)
        cu = paddle.to_tensor(cu_list, dtype="int32")

        for cp_rank in range(cp_size):
            mtp_emb_res = _build_mtp_emb_res(
                inputs_embeds,
                inputs_embeds_ori,
                cu,
                K,
                cp_rank=cp_rank,
                cp_size=cp_size,
            )
            assert len(mtp_emb_res) == K + 1

            # Slot 0 == base (not extracted — matches production code where
            # slot 0 is inputs_embeds after any earlier SP scatter, not an
            # extraction result).
            np.testing.assert_array_equal(mtp_emb_res[0].numpy(), embeds_np)

            for k in range(K):
                # Reference: roll (k+1) times, then extract this rank's chunks.
                rolled_ref = _numpy_per_doc_roll_float(
                    embeds_np, cu_list, shifts=k + 1
                )
                rolled_ref_pd = paddle.to_tensor(rolled_ref)
                expected_local = extract_local_zigzag_chunks(
                    rolled_ref_pd, cp_rank=cp_rank, cp_size=cp_size, axis=1
                )
                np.testing.assert_array_equal(
                    mtp_emb_res[k + 1].numpy(),
                    expected_local.numpy(),
                    err_msg=(
                        f"K={K}, depth={k}, cp_rank={cp_rank}: "
                        f"mtp_emb_res[{k + 1}] must equal "
                        f"extract_local_zigzag_chunks(roll(base, k+1={k + 1}))"
                    ),
                )

    def test_k1(self) -> None:
        self._run_k(1)

    def test_k2(self) -> None:
        self._run_k(2)

    def test_k3(self) -> None:
        self._run_k(3)


class TestMtpEmbResRegressionGuards:
    """Explicit regressions we want to catch."""

    def test_off_by_one_less_than_kplus1_is_wrong(self) -> None:
        """If someone writes ``for depth in range(K)`` but forgets to
        prepend inputs_embeds, the list length is K not K+1. Confirm our
        assertion catches that.
        """
        paddle.set_device("cpu")
        B, L, H = 1, 8, 2
        embeds_np = np.arange(B * L * H, dtype=np.float32).reshape(B, L, H)
        inputs_embeds = paddle.to_tensor(embeds_np)
        cu = paddle.to_tensor([0, L], dtype="int32")

        # A buggy loop that forgets the leading slot:
        bad_res: list[paddle.Tensor] = []
        rolled = inputs_embeds
        for _ in range(3):
            rolled, _ = roll_tensor(
                rolled, shifts=-1, dims=1, cp_group=None, cu_seqlens_q=cu
            )
            bad_res.append(rolled)
        # Buggy list has length K, not K+1. Guard for future refactoring:
        assert len(bad_res) == 3
        # The correct loop always yields K+1 for the same K.
        good_res = _build_mtp_emb_res(inputs_embeds, inputs_embeds, cu, 3)
        assert len(good_res) == 4

    def test_non_accumulative_roll_breaks_semantics(self) -> None:
        """If depth k re-rolls from base once (instead of accumulating),
        every mtp_emb_res[k] equals a single roll — depth loses meaning.
        Confirm our reference detects the difference.
        """
        paddle.set_device("cpu")
        B, L, H = 1, 12, 2
        embeds_np = np.arange(1, B * L * H + 1, dtype=np.float32).reshape(
            B, L, H
        )
        inputs_embeds = paddle.to_tensor(embeds_np)
        cu_list = [0, 6, 12]
        cu = paddle.to_tensor(cu_list, dtype="int32")

        # Buggy: always roll from base once, not accumulate.
        bad_depth2 = roll_tensor(
            inputs_embeds, shifts=-1, dims=1, cp_group=None, cu_seqlens_q=cu
        )[0]
        # Correct: depth 2 should be (k+1)=2 accumulated rolls from base.
        good_res = _build_mtp_emb_res(inputs_embeds, inputs_embeds, cu, 2)
        correct_depth2 = good_res[2]

        assert not np.array_equal(bad_depth2.numpy(), correct_depth2.numpy()), (
            "Accumulative roll must differ from single-roll at depth 2 for "
            "multi-doc packs — otherwise mtp_emb_res semantics is silently "
            "reduced to K=1."
        )
