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

"""Single-card coverage for GPTEmbedding.forward under use_erndata=True.

Drives the REAL ``GPTEmbedding.forward`` via ``__new__`` + MagicMock config
and a stubbed ``embedding`` so the megatron MTP embedding branch runs on a
single GPU (CP=1, SP off):

- cu_seqlens_q ingest + host->GPU move + LanguageLoss stash
- megatron branch entry, multimodal guard, packed roll, moe-mask handling
- CP=1 no-op ``slice_erndata_cp`` for the main embedding and each rolled depth

The CP>1 and sequence_parallel sublines are covered single-card by
monkeypatching ``get_context_parallel_world_size`` -> 2, faking the CP rank,
and replacing ``ScatterOp`` / ``ContextParallelScatterOp`` with identities
(see TestGptEmbeddingMegatronCPSP / TestGptEmbeddingErnie5CPSP).
"""

from __future__ import annotations

import contextlib
import types
import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import paddle

import paddlefleet.models.gpt.gpt_embedding as ge
from paddlefleet.context_parallel_utils import extract_local_cp_chunks
from paddlefleet.models.common.language_loss.language_loss import LanguageLoss
from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding


def _make_embedding(
    K,
    B,
    L,
    H,
    *,
    use_erndata=True,
    magic_send=False,
    cp_balance_mode="dualchunk_allgather",
    mtp_load_weight_only=False,
):
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
    cfg.mtp_load_weight_only = mtp_load_weight_only
    cfg.use_erndata = use_erndata
    cfg.enable_mtp_magic_send = magic_send
    # MagicMock attributes are truthy by default; pin the real default so the
    # mtp_emb_res tail keeps concatenating into hidden_states instead of taking
    # the separate_mtp_input transport (which is a PP=1/K=1 optimization and is
    # rejected for use_erndata=True).
    cfg.separate_mtp_input = False
    cfg.pad_token_id = 0
    cfg.experimental_dataflow = False
    cfg.apply_rope_fusion = False
    cfg.cp_balance_mode = cp_balance_mode
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
    # astype line.
    _stub_embedding.embed_tokens = types.SimpleNamespace(
        weight=types.SimpleNamespace(dtype=paddle.float32)
    )
    emb.embedding = _stub_embedding
    return emb


def _erndata_args(input_ids, cu_seqlens_q=None, **extra):
    """Batch dict for use_erndata=True: adapter-owned flashmask is required."""
    batch, seq = input_ids.shape
    args = {
        "input_ids": input_ids,
        "attn_mask_startend_row_indices": paddle.full(
            [batch, 1, seq, 1], seq, dtype="int32"
        ),
    }
    if cu_seqlens_q is not None:
        args["cu_seqlens_q"] = cu_seqlens_q
    args.update(extra)
    return args


@contextlib.contextmanager
def _fake_cp(cp_size=2, cp_rank=0):
    """Force CP world size / rank so the plain-path CP>1 branch and
    ``slice_erndata_cp`` both see the fake group. The extract helper is left
    REAL (pure slicing); feed a seq length divisible by 2*cp_size.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                ge, "get_context_parallel_world_size", lambda: cp_size
            )
        )
        stack.enter_context(
            mock.patch(
                "paddlefleet.parallel_state.get_context_parallel_world_size",
                lambda: cp_size,
            )
        )
        stack.enter_context(
            mock.patch(
                "paddlefleet.parallel_state.get_context_parallel_rank",
                lambda: cp_rank,
            )
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

        out = emb.forward(_erndata_args(input_ids, cu_cpu))

        # hidden_states is the concat of (K+1) [B, L, H] embeddings along axis 0.
        self.assertEqual(list(out["hidden_states"].shape), [(K + 1) * B, L, H])
        # cu_seqlens_q rides the pipeline dict as a raw tensor, now on GPU.
        self.assertIn("cu_seqlens_q", out)
        self.assertTrue(out["cu_seqlens_q"].place.is_gpu_place())
        # LanguageLoss stash was populated for the loss stage.
        self.assertIsNotNone(LanguageLoss._cu_seqlens_q_stash)

    def test_megatron_magic_send_keeps_one_carrier_and_metadata(self) -> None:
        K, B, L, H = 2, 1, 8, 4
        emb = _make_embedding(K, B, L, H, magic_send=True)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        cu = paddle.to_tensor([0, 3, 8], dtype="int32")

        out = emb.forward(_erndata_args(input_ids, cu))

        self.assertEqual(list(out["hidden_states"].shape), [B, L, H])
        self.assertIn("mtp_full_input_ids", out)
        self.assertTrue(out["mtp_full_input_ids"].is_contiguous())
        self.assertTrue(out["mtp_full_input_ids"].stop_gradient)
        np.testing.assert_array_equal(
            out["mtp_full_input_ids"].numpy(), input_ids.numpy()
        )
        self.assertIn("cu_seqlens_q", out)

    def test_stash_is_gpu_tensor(self) -> None:
        K, B, L, H = 1, 1, 6, 4
        emb = _make_embedding(K, B, L, H)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        cu_cpu = paddle.to_tensor(
            [0, 6], dtype="int32", place=paddle.CPUPlace()
        )
        emb.forward(_erndata_args(input_ids, cu_cpu))
        self.assertTrue(LanguageLoss._cu_seqlens_q_stash.place.is_gpu_place())

    def test_erndata_missing_mask_raises(self) -> None:
        # GPTEmbedding fail-closes for every use_erndata spelling, including
        # the plain path where MTP is inactive (K==0 / weight-only). MTP-layer
        # reuse of a missing mask is covered in test_mtp_forward_dispatch.
        B, L, H = 1, 8, 4
        cases = (
            (0, False),
            (1, False),
            (1, True),
        )
        for k, weight_only in cases:
            with self.subTest(
                num_nextn_predict_layers=k, mtp_load_weight_only=weight_only
            ):
                emb = _make_embedding(
                    k, B, L, H, mtp_load_weight_only=weight_only
                )
                input_ids = (
                    paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
                )
                cu = paddle.to_tensor([0, L], dtype="int32")
                with self.assertRaisesRegex(
                    RuntimeError, r"attn_mask_startend_row_indices"
                ):
                    emb.forward({"input_ids": input_ids, "cu_seqlens_q": cu})

    def test_erndata_plain_k0_cp_slices_locally(self) -> None:
        # K == 0 erndata plain path: with no MTP layer and experimental_dataflow
        # off, forward must slice decoder_input to this rank's local CP chunk
        # via slice_erndata_cp -- the same helper (and cp_balance_mode layout)
        # the MTP branch uses -- and emit no MTP concat/tail.
        B, L, H = 1, 8, 4  # L divisible by 2*cp_size
        for mode in ("dualchunk_allgather", "contiguous_allgather"):
            for cp_rank in (0, 1):
                with self.subTest(cp_balance_mode=mode, cp_rank=cp_rank):
                    emb = _make_embedding(0, B, L, H, cp_balance_mode=mode)
                    # Skip fill_feature (which zeros pad_token positions) and the
                    # experimental-version rope-nulling so hidden_states is the
                    # bare sliced stub embedding, comparable element-for-element.
                    emb.config.gpt_model_use_experimental_version = False
                    input_ids = (
                        paddle.arange(B * L, dtype="int64")
                        .reshape([B, L])
                        .cuda()
                    )
                    cu = paddle.to_tensor([0, L], dtype="int32")
                    full = paddle.arange(B * L * H, dtype="float32").reshape(
                        [B, L, H]
                    )
                    with _fake_cp(cp_size=2, cp_rank=cp_rank):
                        out = emb.forward(_erndata_args(input_ids, cu))
                    self.assertEqual(
                        list(out["hidden_states"].shape), [B, L // 2, H]
                    )
                    expected = extract_local_cp_chunks(
                        full, cp_rank, 2, axis=1, mode=mode
                    )
                    np.testing.assert_array_equal(
                        out["hidden_states"].numpy(), expected.numpy()
                    )
                    self.assertNotIn("mtp_decoder_inputs", out)


class TestGptEmbeddingErnie5(unittest.TestCase):
    """ernie5 (non-megatron) MTP embedding path, single-card (no SP/CP).

    Covers the L+K concat-shift branch (minus SP/CP-only sublines) and the
    magic-send truncation branch.
    """

    def test_ernie5_concat_shift_path(self) -> None:
        K, B, L, H = 2, 1, 10, 4
        emb = _make_embedding(K, B, L, H, use_erndata=False)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        out = emb.forward({"input_ids": input_ids})
        # mtp_emb_res holds K+1 chunks each [B, L-K, H], concatenated on axis 0.
        self.assertEqual(
            list(out["hidden_states"].shape), [(K + 1) * B, L - K, H]
        )

    def test_ernie5_magic_send_truncation(self) -> None:
        K, B, L, H = 2, 1, 10, 4
        emb = _make_embedding(K, B, L, H, use_erndata=False, magic_send=True)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        out = emb.forward({"input_ids": input_ids})
        # magic-send truncates the main embedding to [B, L-K, H] and does not
        # build mtp_emb_res, so hidden_states keeps the single backbone slice.
        self.assertEqual(list(out["hidden_states"].shape), [B, L - K, H])


class TestGptEmbeddingMegatronCPSP(unittest.TestCase):
    """CP>1 and sequence_parallel sublines of the megatron branch,
    covered single-card via monkeypatch.
    """

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_megatron_cp_extract(self) -> None:
        # cp_size=2 -> real slice_erndata_cp halves the seq len.
        # L must be divisible by 2*cp_size.
        K, B, L, H = 2, 1, 8, 4
        emb = _make_embedding(K, B, L, H)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        cu = paddle.to_tensor([0, 3, 8], dtype="int32")
        with _fake_cp(cp_size=2):
            out = emb.forward(_erndata_args(input_ids, cu))
        # Each of the K+1 chunks is zigzag-halved to L/2 on axis 1.
        self.assertEqual(
            list(out["hidden_states"].shape), [(K + 1) * B, L // 2, H]
        )

    def test_megatron_cp_slice_follows_cp_balance_mode(self) -> None:
        # The defect this change fixes: the megatron branch hard-coded the zigzag
        # slice instead of reading cp_balance_mode, so a contiguous_allgather
        # model got embeddings belonging to other ranks' tokens -- a silently
        # wrong loss, not a crash.
        #
        # test_megatron_cp_extract above only asserts the *shape*, which both
        # layouts share, so this is the only assertion that fails when the mode
        # is ignored: it pins values at a rank where the layouts disagree.
        K, B, L, H = 2, 1, 8, 4
        cp_size, cp_rank = 2, 1
        full = paddle.arange(B * L * H, dtype="float32").reshape([B, L, H])
        sliced = {}
        for mode in ("dualchunk_allgather", "contiguous_allgather"):
            emb = _make_embedding(K, B, L, H, cp_balance_mode=mode)
            input_ids = (
                paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
            )
            cu = paddle.to_tensor([0, 3, 8], dtype="int32")
            with _fake_cp(cp_size=cp_size, cp_rank=cp_rank):
                out = emb.forward(_erndata_args(input_ids, cu))
            # hidden_states is the (K+1) chunks concatenated on axis 0; the
            # first B rows are the main (unrolled) embedding.
            main = out["hidden_states"][:B]
            expected = extract_local_cp_chunks(
                full, cp_rank, cp_size, axis=1, mode=mode
            )
            np.testing.assert_array_equal(main.numpy(), expected.numpy())
            sliced[mode] = main.numpy()

        # Guard the guard: if the two layouts happened to agree at this
        # (L, cp_size, cp_rank) the loop above would pass with the mode ignored.
        self.assertFalse(
            np.array_equal(
                sliced["dualchunk_allgather"], sliced["contiguous_allgather"]
            ),
            "the two CP layouts must differ here or this test proves nothing",
        )

    def test_plain_cp_rejects_sequence_parallel(self) -> None:
        # Config already rejects this combo; the runtime check is the
        # -O-safe belt. It must name scatter vs local slice and the
        # inactive-MTP flags, including weight-only (not "no MTP").
        B, L, H = 1, 8, 4
        cases = (
            (0, False, False, "a local CP slice"),
            (0, False, True, "ContextParallelScatterOp"),
            (1, True, False, "a local CP slice"),
        )
        for k, weight_only, dataflow, how in cases:
            with self.subTest(
                num_nextn_predict_layers=k,
                mtp_load_weight_only=weight_only,
                experimental_dataflow=dataflow,
            ):
                emb = _make_embedding(
                    k, B, L, H, mtp_load_weight_only=weight_only
                )
                emb.config.experimental_dataflow = dataflow
                emb.sequence_parallel = True
                emb.config.sequence_parallel = True
                input_ids = (
                    paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
                )
                cu = paddle.to_tensor([0, L], dtype="int32")
                with (
                    _fake_cp(cp_size=2),
                    self.assertRaisesRegex(
                        ValueError,
                        rf"sequence_parallel=True.*{how}"
                        rf".*context_parallel_size=2"
                        rf".*num_nextn_predict_layers={k}"
                        rf".*mtp_load_weight_only={weight_only}"
                        r".*Set sequence_parallel=False",
                    ),
                ):
                    emb.forward(_erndata_args(input_ids, cu))

    def test_megatron_sequence_parallel(self) -> None:
        # Identity ScatterOp still validates the sequence-first layout for B > 1.
        K, B, L, H = 2, 2, 8, 4
        emb = _make_embedding(K, B, L, H)
        emb.sequence_parallel = True
        emb.config.sequence_parallel = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        cu = paddle.to_tensor([0, 3, 8, 11, 16], dtype="int32")
        with _identity_scatter():
            out = emb.forward(_erndata_args(input_ids, cu))
        # SP layout is [S, B, H]; concat of K+1 chunks -> [(K+1)*L, B, H].
        self.assertEqual(list(out["hidden_states"].shape), [(K + 1) * L, B, H])
        expected_main = emb.embedding(input_ids=input_ids, position_ids=None)
        expected_main[0, 0, :] = 0  # pad_token_id=0 is zeroed by GPTEmbedding.
        np.testing.assert_allclose(
            out["hidden_states"][:L].numpy(),
            expected_main.transpose([1, 0, 2]).numpy(),
        )


class TestGptEmbeddingErnie5CPSP(unittest.TestCase):
    """CP scatter and sequence_parallel sublines of the ernie5
    (non-megatron) MTP embedding branch.
    """

    def test_ernie5_cp_scatter(self) -> None:
        # experimental_dataflow + cp_size>1 -> ContextParallelScatterOp
        # (identity).
        K, B, L, H = 2, 1, 10, 4
        emb = _make_embedding(K, B, L, H, use_erndata=False)
        emb.config.experimental_dataflow = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        with _fake_cp(cp_size=2), _identity_scatter():
            out = emb.forward({"input_ids": input_ids})
        self.assertEqual(
            list(out["hidden_states"].shape), [(K + 1) * B, L - K, H]
        )

    def test_ernie5_sequence_parallel(self) -> None:
        # sequence_parallel path with identity ScatterOp.
        K, B, L, H = 2, 1, 10, 4
        emb = _make_embedding(K, B, L, H, use_erndata=False)
        emb.sequence_parallel = True
        emb.config.sequence_parallel = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        with _identity_scatter():
            out = emb.forward({"input_ids": input_ids})
        # SP layout [S, B, H]; concat of K+1 chunks each [L-K, B, H].
        self.assertEqual(
            list(out["hidden_states"].shape), [(K + 1) * (L - K), B, H]
        )


class _StubRope:
    """Stand-in for ``RotaryEmbedding`` with an assertable layout.

    Mirrors the two behaviours that matter here:
    ``get_rotary_seq_len`` scales the rank-local length back up by the CP world
    size (so the table is always built for the FULL length L), and ``__call__``
    returns a ``[1, S, 1, D]`` table. Every channel holds the *global* position
    index, which makes the CP slicing directly comparable against
    ``extract_local_zigzag_chunks``.
    """

    def __init__(self, cp_size, dim=4):
        self.cp_size = cp_size
        self.dim = dim

    def get_rotary_seq_len(
        self, transformer_input, config, packed_seq_params=None
    ):
        return transformer_input.shape[1] * self.cp_size

    def __call__(self, seq_len, packed_seq=False, position_ids=None):
        pos = paddle.arange(seq_len, dtype="float32")
        return pos.reshape([1, seq_len, 1, 1]).tile([1, 1, 1, self.dim])


def _enable_rope(emb, cp_size, dim=4):
    """Switch a fake embedding onto the real RoPE codepath."""
    emb.position_embedding_type = "rope"
    emb.rotary_pos_emb = _StubRope(cp_size, dim)
    emb.training = True
    # gpt_model_use_experimental_version nulls out every rope tensor.
    emb.config.gpt_model_use_experimental_version = False
    emb.config.apply_rope_fusion = False
    return emb


class TestGptEmbeddingMegatronCPRope(unittest.TestCase):
    """RoPE must follow the megatron branch's zigzag CP layout.

    ``ContextParallelScatterOp`` is gated on ``experimental_dataflow``, which
    use_erndata=True forbids, so without an explicit slice the
    full-length table would reach a decoder that only holds L/cp positions.
    """

    def setUp(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def tearDown(self) -> None:
        LanguageLoss._cu_seqlens_q_stash = None

    def test_rope_is_zigzag_sliced(self) -> None:
        from paddlefleet.context_parallel_utils import (
            extract_local_zigzag_chunks,
        )

        K, B, L, H, D = 2, 1, 8, 4, 4
        emb = _enable_rope(_make_embedding(K, B, L, H), cp_size=2, dim=D)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        cu = paddle.to_tensor([0, 3, 8], dtype="int32")
        with _fake_cp(cp_size=2):
            out = emb.forward(_erndata_args(input_ids, cu))

        full = (
            paddle.arange(L, dtype="float32")
            .reshape([1, L, 1, 1])
            .tile([1, 1, 1, D])
        )
        expected = extract_local_zigzag_chunks(full, 0, 2, axis=1)
        np.testing.assert_array_equal(
            out["rotary_pos_emb"].numpy(), expected.numpy()
        )
        # cp_rank 0 of cp_size 2 owns interval=L/4=2 -> positions [0, 1, 6, 7].
        np.testing.assert_array_equal(
            out["rotary_pos_emb"][0, :, 0, 0].numpy(),
            np.array([0, 1, 6, 7], dtype="float32"),
        )
        # Matches the hidden-state length so the decoder can consume it.
        self.assertEqual(
            out["rotary_pos_emb"].shape[1], out["hidden_states"].shape[1]
        )

    def test_rope_slice_follows_cp_balance_mode(self) -> None:
        # RoPE slicing is a second, independent call site of slice_erndata_cp.
        # Every channel of the stub table holds the
        # global position index, so the sliced table reads out directly as the
        # position set this rank owns -- and the two layouts disagree at rank 1
        # of 2 with L=8 (zigzag [2,3,4,5] vs contiguous [4,5,6,7]).
        K, B, L, H, D = 2, 1, 8, 4, 4
        expected_positions = {
            "dualchunk_allgather": [2, 3, 4, 5],
            "contiguous_allgather": [4, 5, 6, 7],
        }
        for mode, positions in expected_positions.items():
            emb = _enable_rope(
                _make_embedding(K, B, L, H, cp_balance_mode=mode),
                cp_size=2,
                dim=D,
            )
            input_ids = (
                paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
            )
            cu = paddle.to_tensor([0, 3, 8], dtype="int32")
            with _fake_cp(cp_size=2, cp_rank=1):
                out = emb.forward(_erndata_args(input_ids, cu))
            np.testing.assert_array_equal(
                out["rotary_pos_emb"][0, :, 0, 0].numpy(),
                np.array(positions, dtype="float32"),
                err_msg=f"rope slice ignored cp_balance_mode={mode!r}",
            )

    def test_rope_untouched_without_cp(self) -> None:
        K, B, L, H, D = 2, 1, 8, 4, 4
        emb = _enable_rope(_make_embedding(K, B, L, H), cp_size=1, dim=D)
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        cu = paddle.to_tensor([0, 3, 8], dtype="int32")
        out = emb.forward(_erndata_args(input_ids, cu))
        np.testing.assert_array_equal(
            out["rotary_pos_emb"][0, :, 0, 0].numpy(),
            np.arange(L, dtype="float32"),
        )

    def test_ernie5_rope_not_sliced(self) -> None:
        """Regression guard: the slice must be megatron-only."""
        K, B, L, H, D = 2, 1, 8, 4, 4
        # cp_size=2 makes the stub scale the rank-local length (L - K) back up
        # to the full 2*(L - K), mirroring the real get_rotary_seq_len.
        emb = _enable_rope(
            _make_embedding(K, B, L, H, use_erndata=False),
            cp_size=2,
            dim=D,
        )
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        with _fake_cp(cp_size=2):
            out = emb.forward({"input_ids": input_ids})
        # ernie5 backbone length is L - K and its own CP scatter is gated on
        # experimental_dataflow, so the table stays full-length as generated.
        np.testing.assert_array_equal(
            out["rotary_pos_emb"][0, :, 0, 0].numpy(),
            np.arange((L - K) * 2, dtype="float32"),
        )


class TestGptEmbeddingMagicSendCPSP(unittest.TestCase):
    """magic-send truncation branch CP/SP sublines. magic-send lives in the
    ernie5 (non-megatron) path, so use_erndata stays False.
    """

    def test_magic_experimental_sp_cp(self) -> None:
        # experimental_version=True + SP=True + CP>1 + experimental_dataflow.
        K, B, L, H = 2, 1, 8, 4
        emb = _make_embedding(K, B, L, H, use_erndata=False, magic_send=True)
        emb.sequence_parallel = True
        emb.config.sequence_parallel = True
        emb.config.experimental_dataflow = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        with _fake_cp(cp_size=2), _identity_scatter():
            out = emb.forward({"input_ids": input_ids})
        self.assertIn("hidden_states", out)

    def test_magic_non_experimental_sp(self) -> None:
        # experimental_version=False + SP=True: the ``if not (experimental
        # and SP)`` guard is True, so the reshape/permute runs.
        K, B, L, H = 2, 1, 8, 4
        emb = _make_embedding(K, B, L, H, use_erndata=False, magic_send=True)
        emb.config.gpt_model_use_experimental_version = False
        emb.sequence_parallel = True
        emb.config.sequence_parallel = True
        input_ids = paddle.arange(B * L, dtype="int64").reshape([B, L]).cuda()
        with _identity_scatter():
            out = emb.forward({"input_ids": input_ids})
        self.assertIn("hidden_states", out)


if __name__ == "__main__":
    unittest.main()
