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

"""Single-card coverage for GPTEmbedding.forward under mtp_data_style="megatron".

Drives the REAL ``GPTEmbedding.forward`` via ``__new__`` + MagicMock config
and a stubbed ``embedding`` so the megatron MTP embedding branch runs on a
single GPU (CP=1, SP off):

- cu_seqlens_q ingest + host->GPU move + LanguageLoss stash
  (gpt_embedding.py:345-346, 355-360).
- megatron branch entry, multimodal guard, roll import, moe-mask handling
  (lines 430-447).
- CP=1 no-op extract for the main embedding and each rolled depth
  (lines 450-451, 456/461, 483-498, 513).

The CP>1 and sequence_parallel sublines (457,494 / 464,467,501-502,505,508
for megatron; 603,610,613-614,638,647,650,653 for ernie5) are covered
single-card by monkeypatching ``get_context_parallel_world_size`` -> 2,
faking the CP rank, and replacing ``ScatterOp`` / ``ContextParallelScatterOp``
with identities (see TestGptEmbeddingMegatronCPSP / TestGptEmbeddingErnie5CPSP).
"""

from __future__ import annotations

import contextlib
import types
import unittest
from unittest import mock
from unittest.mock import MagicMock

import paddle

import paddlefleet.models.gpt.gpt_embedding as ge
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding


def _make_embedding(K, B, L, H, *, mtp_data_style="megatron", magic_send=False):
    emb = GPTEmbedding.__new__(GPTEmbedding)
    cfg = MagicMock()
    cfg.gpt_model_use_experimental_version = True
    cfg.max_sequence_length = 128
    cfg.sequence_parallel = False
    cfg.multi_latent_attention = False
    cfg.multimodal_embedding = False
    cfg.expert_model_parallel_size = 1
    cfg.tensor_model_parallel_size = 1
    cfg.num_nextn_predict_layers = K
    cfg.mtp_load_weight_only = False
    cfg.mtp_data_style = mtp_data_style
    cfg.enable_mtp_magic_send = magic_send
    cfg.pad_token_id = 0
    cfg.experimental_dataflow = False
    cfg.apply_rope_fusion = False
    cfg.cp_balance_mode = "zigzag"
    cfg.clone_scatter_output_in_embedding = False
    cfg.layer_types = []  # -> has_kda_layer property returns False
    emb.config = cfg

    emb.multimodal_embedding = False
    emb.sequence_parallel = False
    emb.position_embedding_type = "none"
    emb.rotary_pos_emb = None
    emb.swa_rotary_pos_emb = None

    def _stub_embedding(input_ids, position_ids=None):
        b, s = input_ids.shape
        return paddle.arange(b * s * H, dtype="float32").reshape([b, s, H])

    # embed_tokens.weight.dtype is read by the magic-send experimental+SP
    # astype line (gpt_embedding.py:564).
    _stub_embedding.embed_tokens = types.SimpleNamespace(
        weight=types.SimpleNamespace(dtype=paddle.float32)
    )
    emb.embedding = _stub_embedding
    return emb


@contextlib.contextmanager
def _fake_cp(cp_size=2):
    """Force ``get_context_parallel_world_size`` -> cp_size and rank -> 0 in
    the gpt_embedding namespace so ``if _cp_size > 1`` branches execute.
    ``extract_local_zigzag_chunks`` is left REAL (pure slicing) so shapes stay
    correct; feed a seq length divisible by 2*cp_size.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                ge, "get_context_parallel_world_size", lambda: cp_size
            )
        )
        stack.enter_context(
            mock.patch.object(ge, "get_context_parallel_rank", lambda: 0)
        )
        yield


@contextlib.contextmanager
def _identity_scatter():
    """Replace SP/CP scatter PyLayers with identity so the reshape/scatter
    sublines run without a real TP/CP group.
    """

    def identity(x, *a, **k):
        return x

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(ge.ScatterOp, "apply", identity))
        stack.enter_context(
            mock.patch.object(ge.ContextParallelScatterOp, "apply", identity)
        )
        yield


class TestGptEmbeddingMegatron(unittest.TestCase):
    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_megatron_branch_builds_mtp_concat(self) -> None:
        K, B, L, H = 2, 1, 8, 4
        emb = _make_embedding(K, B, L, H)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        # cu on CPU so the .cuda() move (line 346) is exercised.
        cu_cpu = paddle.to_tensor(
            [0, 3, 8], dtype="int32", place=paddle.CPUPlace()
        )

        out = emb.forward({"input_ids": input_ids, "cu_seqlens_q": cu_cpu})

        # hidden_states is the concat of (K+1) [B, L, H] embeddings along axis 0.
        self.assertEqual(list(out["hidden_states"].shape), [(K + 1) * B, L, H])
        # cu_seqlens_q rides the pipeline dict as a raw tensor, now on GPU.
        self.assertIn("cu_seqlens_q", out)
        self.assertTrue(out["cu_seqlens_q"].place.is_gpu_place())
        # LanguageLoss stash was populated for the loss stage.
        self.assertIsNotNone(LanguageLoss._cu_seqlens_q_stash)

    def test_stash_is_gpu_tensor(self) -> None:
        K, B, L, H = 1, 1, 6, 4
        emb = _make_embedding(K, B, L, H)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        cu_cpu = paddle.to_tensor(
            [0, 6], dtype="int32", place=paddle.CPUPlace()
        )
        emb.forward({"input_ids": input_ids, "cu_seqlens_q": cu_cpu})
        self.assertTrue(LanguageLoss._cu_seqlens_q_stash.place.is_gpu_place())


class TestGptEmbeddingErnie5(unittest.TestCase):
    """ernie5 (non-megatron) MTP embedding path, single-card (no SP/CP).

    Covers the L+K concat-shift branch (gpt_embedding.py:516-540, 586-660
    minus SP/CP-only sublines) and the magic-send truncation branch
    (542-545, 551/560/567 guard evaluations).
    """

    def test_ernie5_concat_shift_path(self) -> None:
        K, B, L, H = 2, 1, 10, 4
        emb = _make_embedding(K, B, L, H, mtp_data_style="ernie5")
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        out = emb.forward({"input_ids": input_ids})
        # mtp_emb_res holds K+1 chunks each [B, L-K, H], concatenated on axis 0.
        self.assertEqual(
            list(out["hidden_states"].shape), [(K + 1) * B, L - K, H]
        )

    def test_ernie5_magic_send_truncation(self) -> None:
        K, B, L, H = 2, 1, 10, 4
        emb = _make_embedding(
            K, B, L, H, mtp_data_style="ernie5", magic_send=True
        )
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        out = emb.forward({"input_ids": input_ids})
        # magic-send truncates the main embedding to [B, L-K, H] and does not
        # build mtp_emb_res, so hidden_states keeps the single backbone slice.
        self.assertEqual(list(out["hidden_states"].shape), [B, L - K, H])


class TestGptEmbeddingMegatronCPSP(unittest.TestCase):
    """CP>1 (457, 494) and sequence_parallel (464, 467, 501-502, 505, 508)
    sublines of the megatron branch, covered single-card via monkeypatch.
    """

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_megatron_cp_extract(self) -> None:
        # cp_size=2 -> real extract_local_zigzag_chunks halves the seq len
        # (lines 457 and 494). L must be divisible by 2*cp_size.
        K, B, L, H = 2, 1, 8, 4
        emb = _make_embedding(K, B, L, H)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        cu = paddle.to_tensor([0, 3, 8], dtype="int32")
        with _fake_cp(cp_size=2):
            out = emb.forward({"input_ids": input_ids, "cu_seqlens_q": cu})
        # Each of the K+1 chunks is zigzag-halved to L/2 on axis 1.
        self.assertEqual(
            list(out["hidden_states"].shape), [(K + 1) * B, L // 2, H]
        )

    def test_megatron_sequence_parallel(self) -> None:
        # sequence_parallel path with an identity ScatterOp
        # (lines 464, 467, 501-502, 505, 508).
        K, B, L, H = 2, 1, 8, 4
        emb = _make_embedding(K, B, L, H)
        emb.sequence_parallel = True
        emb.config.sequence_parallel = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        cu = paddle.to_tensor([0, 3, 8], dtype="int32")
        with _identity_scatter():
            out = emb.forward({"input_ids": input_ids, "cu_seqlens_q": cu})
        # SP layout is [S, B, H]; concat of K+1 chunks -> [(K+1)*L, B, H].
        self.assertEqual(list(out["hidden_states"].shape), [(K + 1) * L, B, H])


class TestGptEmbeddingErnie5CPSP(unittest.TestCase):
    """CP scatter (603, 638) and sequence_parallel (610, 613-614, 647, 650,
    653) sublines of the ernie5 (non-megatron) MTP embedding branch.
    """

    def test_ernie5_cp_scatter(self) -> None:
        # experimental_dataflow + cp_size>1 -> ContextParallelScatterOp
        # (identity) at lines 603 and 638.
        K, B, L, H = 2, 1, 10, 4
        emb = _make_embedding(K, B, L, H, mtp_data_style="ernie5")
        emb.config.experimental_dataflow = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        with _fake_cp(cp_size=2), _identity_scatter():
            out = emb.forward({"input_ids": input_ids})
        self.assertEqual(
            list(out["hidden_states"].shape), [(K + 1) * B, L - K, H]
        )

    def test_ernie5_sequence_parallel(self) -> None:
        # sequence_parallel path with identity ScatterOp
        # (lines 610, 613-614, 647, 650, 653).
        K, B, L, H = 2, 1, 10, 4
        emb = _make_embedding(K, B, L, H, mtp_data_style="ernie5")
        emb.sequence_parallel = True
        emb.config.sequence_parallel = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        with _identity_scatter():
            out = emb.forward({"input_ids": input_ids})
        # SP layout [S, B, H]; concat of K+1 chunks each [L-K, B, H].
        self.assertEqual(
            list(out["hidden_states"].shape), [(K + 1) * (L - K), B, H]
        )


class TestGptEmbeddingMagicSendCPSP(unittest.TestCase):
    """magic-send truncation branch CP/SP sublines (gpt_embedding.py:555,
    564, 568, 571, 574-575, 579). magic-send lives in the ernie5
    (non-megatron) path, so mtp_data_style stays "ernie5".
    """

    def test_magic_experimental_sp_cp(self) -> None:
        # experimental_version=True + SP=True + CP>1 + experimental_dataflow:
        # covers 555 (CP scatter), 564 (astype), 568/571/574 (SP reshape),
        # 575 (guard eval). 579 is skipped here (experimental&SP True).
        K, B, L, H = 2, 1, 8, 4
        emb = _make_embedding(
            K, B, L, H, mtp_data_style="ernie5", magic_send=True
        )
        emb.sequence_parallel = True
        emb.config.sequence_parallel = True
        emb.config.experimental_dataflow = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        with _fake_cp(cp_size=2), _identity_scatter():
            out = emb.forward({"input_ids": input_ids})
        self.assertIn("hidden_states", out)

    def test_magic_non_experimental_sp(self) -> None:
        # experimental_version=False + SP=True: the ``if not (experimental
        # and SP)`` guard is True, so line 579 (reshape/permute) runs.
        K, B, L, H = 2, 1, 8, 4
        emb = _make_embedding(
            K, B, L, H, mtp_data_style="ernie5", magic_send=True
        )
        emb.config.gpt_model_use_experimental_version = False
        emb.sequence_parallel = True
        emb.config.sequence_parallel = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        with _identity_scatter():
            out = emb.forward({"input_ids": input_ids})
        self.assertIn("hidden_states", out)


if __name__ == "__main__":
    unittest.main()
