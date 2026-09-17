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

"""erndata MTP contracts: cu_seqlens granularity + mHC multi-depth chain.

Two contracts that end-to-end smoke can only surface as opaque shape errors
(or, worse, as a silently wrong attention mask):

1. ``build_startend_row_indices_from_cu_seqlens`` must decide the cu_seqlens
   granularity from the caller-supplied per-sample ``seq_len``, never from the
   boundary values. ``batch_size=2`` with per-sample ``cu=[0, 4, 8]`` has the
   same boundary set as a batch-flat cu over two length-4 samples, so a
   structural test necessarily mis-classifies one of them and emits a mask of
   the wrong length.

2. Under ``use_erndata=True`` with mHC, ``_forward_megatron_style`` must
   publish each depth's multi-stream output into the ``mhc_multistream``
   channel. Leaving the backbone-generated chunk in place makes depth k+1 read
   a stale slot, so for K > 1 the MTP multi-stream chain is no longer serial.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import paddle

from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    build_startend_row_indices_from_cu_seqlens,
    roll_tensor,
)


def _ends(mask: paddle.Tensor, sample: int) -> list[int]:
    """Extract the per-position end-row column for one sample."""
    return [row[0] for row in mask.numpy().tolist()[sample][0]]


class TestCuSeqlensGranularity(unittest.TestCase):
    """seq_len decides the granularity; boundary values never do."""

    def test_per_sample_cu_colliding_with_flat_layout(self) -> None:
        """cu=[0,4,8] with batch_size=2 is per-sample when seq_len=8.

        The boundary 4 makes this indistinguishable from a batch-flat cu over
        two length-4 samples. Passing the real seq_len must keep the per-sample
        reading: one shared doc layout of length 8, broadcast to both samples.
        A flat misreading would emit length-4 rows and break attention.
        """
        mask = build_startend_row_indices_from_cu_seqlens(
            paddle.to_tensor([0, 4, 8], dtype="int32"),
            batch_size=2,
            seq_len=8,
        )
        self.assertEqual(list(mask.shape), [2, 1, 8, 1])
        expected = [4, 4, 4, 4, 8, 8, 8, 8]
        self.assertEqual(_ends(mask, 0), expected)
        self.assertEqual(_ends(mask, 1), expected)

    def test_batch_flat_cu_materializes_per_sample_boundaries(self) -> None:
        """cu spanning batch*seq_len gets per-sample doc boundaries.

        batch_size=2, seq_len=4 -> span 8. Sample 0 owns [0,4) with docs
        [0,2) and [2,4); sample 1 owns [4,8) with docs [4,7) and [7,8), which
        map to local ends 3 and 4. The two samples must NOT share a layout.
        """
        mask = build_startend_row_indices_from_cu_seqlens(
            paddle.to_tensor([0, 2, 4, 7, 8], dtype="int32"),
            batch_size=2,
            seq_len=4,
        )
        self.assertEqual(list(mask.shape), [2, 1, 4, 1])
        self.assertEqual(_ends(mask, 0), [2, 2, 4, 4])
        self.assertEqual(_ends(mask, 1), [3, 3, 3, 4])

    def test_flat_and_per_sample_cu_differ_for_same_boundaries(self) -> None:
        """Same boundary set, different seq_len -> different masks.

        This is the collision the structural detection could not resolve:
        cu=[0,4,8] is a length-8 per-sample layout when seq_len=8 and a
        batch-flat layout over two length-4 samples when seq_len=4.
        """
        cu = paddle.to_tensor([0, 4, 8], dtype="int32")
        per_sample = build_startend_row_indices_from_cu_seqlens(
            cu, batch_size=2, seq_len=8
        )
        flat = build_startend_row_indices_from_cu_seqlens(
            cu, batch_size=2, seq_len=4
        )
        self.assertEqual(list(per_sample.shape), [2, 1, 8, 1])
        self.assertEqual(list(flat.shape), [2, 1, 4, 1])
        # Flat: each sample is one full doc, so every end is the local length.
        self.assertEqual(_ends(flat, 0), [4, 4, 4, 4])
        self.assertEqual(_ends(flat, 1), [4, 4, 4, 4])

    def test_include_position_axis_flat(self) -> None:
        """The 2-column experimental layout carries per-sample positions."""
        mask = build_startend_row_indices_from_cu_seqlens(
            paddle.to_tensor([0, 2, 4, 8], dtype="int32"),
            batch_size=2,
            seq_len=4,
            include_position_axis=True,
        )
        self.assertEqual(list(mask.shape), [2, 1, 4, 2])
        for sample in range(2):
            pos = [row[1] for row in mask.numpy().tolist()[sample][0]]
            self.assertEqual(pos, [0, 1, 2, 3])

    def test_missing_seq_len_keeps_per_sample_semantics(self) -> None:
        """seq_len=None (unknown L) must not guess; stay per-sample."""
        mask = build_startend_row_indices_from_cu_seqlens(
            paddle.to_tensor([0, 4, 8], dtype="int32"), batch_size=2
        )
        self.assertEqual(list(mask.shape), [2, 1, 8, 1])

    def test_span_neither_per_sample_nor_flat_raises(self) -> None:
        """A span that matches neither reading is a wiring bug, not a guess."""
        with self.assertRaises(ValueError):
            build_startend_row_indices_from_cu_seqlens(
                paddle.to_tensor([0, 8, 16], dtype="int32"),
                batch_size=1,
                seq_len=8,
            )

    def test_batch_size_one_flat_is_per_sample(self) -> None:
        """batch_size=1: flat and per-sample coincide; no flat branch."""
        mask = build_startend_row_indices_from_cu_seqlens(
            paddle.to_tensor([0, 3, 8], dtype="int32"),
            batch_size=1,
            seq_len=8,
        )
        self.assertEqual(list(mask.shape), [1, 1, 8, 1])
        self.assertEqual(_ends(mask, 0), [3, 3, 3, 8, 8, 8, 8, 8])


# Multi-stream chunk marker values: chunk i is filled with (i + 1) * 10 so a
# stale slot is immediately visible in the assertions below.
_CHUNK_MARK = 10.0
# The stubbed transformer block adds this so each depth's output is distinct
# from every backbone-generated chunk.
_BLOCK_DELTA = 1.0


def _make_mhc_layer(K: int, layer_number: int, n: int, h: int):
    """Real MultiTokenPredictionLayer with stubbed block + postprocess.

    ``__new__`` avoids fleet init; only the fields ``_forward_megatron_style``
    touches are populated, so the method under test is the production one.
    """
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    cfg = MagicMock()
    cfg.use_erndata = True
    cfg.enable_mtp_magic_send = False
    cfg.num_nextn_predict_layers = K
    cfg.gpt_model_use_experimental_version = False
    cfg.sequence_parallel = False
    cfg.num_residual_streams = n
    cfg.hidden_size = h
    layer.config = cfg
    layer.layer_number = layer_number
    layer.mhc_enabled = True

    recorded = {}

    def _stub_proj(hidden_states, decoder_input, **kwargs):
        # The shared mHC block only accepts multi-stream input. Raise instead
        # of assert so `python -O` cannot strip the check.
        if hidden_states.shape[-1] != n * h:
            raise RuntimeError(
                f"depth {layer_number} received width "
                f"{hidden_states.shape[-1]}, expected multi-stream {n * h}"
            )
        recorded["hidden_in"] = hidden_states
        recorded["decoder_in"] = decoder_input
        return hidden_states + _BLOCK_DELTA

    def _stub_postprocess(hidden_states):
        # Stand-in for learned_output_contract: [.., n*h] -> [.., h].
        return hidden_states[..., :h]

    layer._proj_and_transformer_layer = _stub_proj
    layer._postprocess = _stub_postprocess
    return layer, recorded


class TestErndataMhcMultiDepthChain(unittest.TestCase):
    """K=2: depth 1 must consume depth 0's multi-stream output."""

    def setUp(self) -> None:
        self.K, self.B, self.S, self.n, self.h = 2, 1, 4, 2, 2

    def _initial_args(self) -> dict:
        K, B, S, n, h = self.K, self.B, self.S, self.n, self.h
        # Multi-stream channel from the backbone contract layer:
        # K+1 slots concatenated along the batch axis.
        mhc = paddle.concat(
            [
                paddle.full([B, S, n * h], (i + 1) * _CHUNK_MARK)
                for i in range(K + 1)
            ]
        )
        # Single-stream carrier: per-depth embeddings.
        carrier = paddle.concat(
            [paddle.full([B, S, h], -(i + 1.0)) for i in range(K + 1)]
        )
        return {
            "hidden_states": carrier,
            "mhc_multistream": mhc,
            "context": None,
        }

    def test_depth1_receives_depth0_multistream_output(self) -> None:
        K, B, S, n, h = self.K, self.B, self.S, self.n, self.h
        l0, rec0 = _make_mhc_layer(K, 0, n, h)
        l1, rec1 = _make_mhc_layer(K, 1, n, h)

        args = self._initial_args()
        out0 = l0.forward(args)

        # Depth 0 consumes backbone chunk 0 (marker 10) as multi-stream input
        # and the carrier slot 1 as its decoder embedding.
        self.assertEqual(float(rec0["hidden_in"].numpy()[0, 0, 0]), _CHUNK_MARK)
        self.assertEqual(list(rec0["hidden_in"].shape), [B, S, n * h])
        self.assertEqual(list(rec0["decoder_in"].shape), [B, S, h])

        # The channel must still be present: depth 1 has yet to run.
        self.assertIn("mhc_multistream", out0)

        out1 = l1.forward(out0)

        # The contract under test: depth 1's multi-stream input is depth 0's
        # output (10 + 1), not the stale backbone chunk 1 (20).
        self.assertEqual(
            float(rec1["hidden_in"].numpy()[0, 0, 0]),
            _CHUNK_MARK + _BLOCK_DELTA,
        )
        self.assertNotEqual(
            float(rec1["hidden_in"].numpy()[0, 0, 0]), 2 * _CHUNK_MARK
        )

        # Last depth: the channel is dropped instead of being forwarded.
        self.assertNotIn("mhc_multistream", out1)

        # The carrier stays width-uniform across all K+1 slots.
        self.assertEqual(list(out1["hidden_states"].shape), [(K + 1) * B, S, h])
        self.assertNotIn("decoder_input", out1)

    def test_carrier_slots_hold_contracted_outputs(self) -> None:
        """Each depth writes its contracted output into carrier slot k+1."""
        K, n, h = self.K, self.n, self.h
        l0, _ = _make_mhc_layer(K, 0, n, h)
        l1, _ = _make_mhc_layer(K, 1, n, h)

        out1 = l1.forward(l0.forward(self._initial_args()))
        slots = paddle.split(out1["hidden_states"], K + 1)

        # Slot 0 is untouched backbone carrier; slots 1..K hold the contracted
        # per-depth outputs (marker + delta).
        self.assertEqual(float(slots[0].numpy()[0, 0, 0]), -1.0)
        self.assertEqual(
            float(slots[1].numpy()[0, 0, 0]), _CHUNK_MARK + _BLOCK_DELTA
        )
        self.assertEqual(
            float(slots[2].numpy()[0, 0, 0]),
            _CHUNK_MARK + 2 * _BLOCK_DELTA,
        )

    def test_k1_drops_channel_immediately(self) -> None:
        """K=1: the only depth is the last one, so nothing is forwarded."""
        B, S, n, h = 1, 4, 2, 2
        layer, rec = _make_mhc_layer(1, 0, n, h)
        mhc = paddle.concat(
            [
                paddle.full([B, S, n * h], (i + 1) * _CHUNK_MARK)
                for i in range(2)
            ]
        )
        carrier = paddle.concat(
            [paddle.full([B, S, h], -(i + 1.0)) for i in range(2)]
        )
        out = layer.forward(
            {
                "hidden_states": carrier,
                "mhc_multistream": mhc,
                "context": None,
            }
        )
        self.assertEqual(float(rec["hidden_in"].numpy()[0, 0, 0]), _CHUNK_MARK)
        self.assertNotIn("mhc_multistream", out)
        self.assertEqual(list(out["hidden_states"].shape), [2 * B, S, h])

    def test_single_stream_input_raises_readable_error(self) -> None:
        """Missing mhc_multistream must fail loudly, not as a raw ReshapeOp.

        Without the channel the shared mHC block would be handed the
        single-stream carrier; the stub asserts the width, mirroring the
        explicit check in ``_concat_embeddings``.
        """
        B, S, n, h = 1, 4, 2, 2
        layer, _ = _make_mhc_layer(1, 0, n, h)
        carrier = paddle.concat(
            [paddle.full([B, S, h], -(i + 1.0)) for i in range(2)]
        )
        with self.assertRaises(RuntimeError):
            layer.forward({"hidden_states": carrier, "context": None})


def _make_mask_layer(
    K: int,
    *,
    layer_number: int = 0,
    sequence_parallel: bool = False,
    tp_size: int = 1,
):
    """Non-mHC layer used to exercise the mask-derivation fallback only."""
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    cfg = MagicMock()
    cfg.use_erndata = True
    cfg.enable_mtp_magic_send = False
    cfg.num_nextn_predict_layers = K
    cfg.gpt_model_use_experimental_version = False
    cfg.sequence_parallel = sequence_parallel
    cfg.tensor_model_parallel_size = tp_size
    layer.config = cfg
    layer.layer_number = layer_number
    layer.mhc_enabled = False

    recorded = {}

    def _stub_proj(hidden_states, decoder_input, **kwargs):
        am = kwargs.get("attn_mask_startend_row_indices")
        recorded["attn_mask"] = am
        recorded["attn_mask_shape"] = None if am is None else list(am.shape)
        return decoder_input

    layer._proj_and_transformer_layer = _stub_proj
    return layer, recorded


class TestErndataMagicSend(unittest.TestCase):
    def _make_layer(self, K: int, depth: int, *, magic_send: bool = True):
        layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
        cfg = MagicMock()
        cfg.use_erndata = True
        cfg.enable_mtp_magic_send = magic_send
        cfg.num_nextn_predict_layers = K
        cfg.gpt_model_use_experimental_version = False
        cfg.sequence_parallel = False
        cfg.tensor_model_parallel_size = 1
        cfg.expert_model_parallel_size = 1
        cfg.pad_token_id = 0
        cfg.cp_balance_mode = "dualchunk_allgather"
        layer.config = cfg
        layer.layer_number = depth
        layer.sequence_parallel = False
        layer.mhc_enabled = False
        object.__setattr__(layer, "mtp_embed", paddle.nn.Embedding(16, 1))
        layer.mtp_embed.weight.set_value(
            paddle.arange(16, dtype="float32").reshape([16, 1])
        )
        recorded = {}

        def _stub_proj(hidden_states, decoder_input, **kwargs):
            recorded["hidden"] = hidden_states
            recorded["decoder"] = decoder_input
            recorded["kwargs"] = kwargs
            return hidden_states + 1

        layer._proj_and_transformer_layer = _stub_proj
        return layer, recorded

    def test_multidepth_grows_carrier_and_rolls_float_embedding(self) -> None:
        K, B, S = 2, 1, 6
        ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6]], dtype="int64")
        cu = paddle.to_tensor([0, 3, 6], dtype="int32")
        args = {
            "hidden_states": paddle.zeros([B, S, 1]),
            "mtp_full_input_ids": ids,
            "cu_seqlens_q": cu,
            "context": None,
        }
        l0, rec0 = self._make_layer(K, 0)
        l1, rec1 = self._make_layer(K, 1)
        out0 = l0._forward_megatron_style(args)
        self.assertIn("mtp_full_input_ids", out0)
        self.assertIs(out0["mtp_full_input_ids"], ids)
        self.assertIs(out0["cu_seqlens_q"], cu)
        self.assertNotIn("mtp_full_input_ids", rec0["kwargs"])
        self.assertEqual(list(out0["hidden_states"].shape), [2 * B, S, 1])
        self.assertEqual(
            rec0["decoder"].squeeze(-1).numpy().tolist(),
            [[2.0, 3.0, 0.0, 5.0, 6.0, 0.0]],
        )
        out1 = l1._forward_megatron_style(out0)
        self.assertEqual(list(out1["hidden_states"].shape), [3 * B, S, 1])
        self.assertNotIn("mtp_full_input_ids", out1)
        self.assertIs(out1["cu_seqlens_q"], cu)
        self.assertEqual(
            rec1["decoder"].squeeze(-1).numpy().tolist(),
            [[3.0, 0.0, 0.0, 6.0, 0.0, 0.0]],
        )
        self.assertTrue(paddle.equal_all(rec1["hidden"], rec0["hidden"] + 1))

    def test_ep_padding_is_zeroed_before_packed_roll(self) -> None:
        layer, _ = self._make_layer(1, 0)
        layer.config.expert_model_parallel_size = 2
        layer.config.tensor_model_parallel_size = 1
        layer.mtp_embed.weight.set_value(paddle.ones_like(layer.mtp_embed.weight))
        ids = paddle.to_tensor([[1, 0, 2, 3, 4, 5]], dtype="int64")
        cu = paddle.to_tensor([0, 3, 6], dtype="int32")
        decoder = layer._prepare_erndata_magic_decoder_input(ids, cu)
        self.assertEqual(
            decoder.squeeze(-1).numpy().tolist(),
            [[0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
        )

    def test_experimental_version_padding_is_zeroed_before_roll(self) -> None:
        layer, _ = self._make_layer(1, 0)
        layer.config.expert_model_parallel_size = 1
        layer.config.tensor_model_parallel_size = 1
        layer.config.gpt_model_use_experimental_version = True
        layer.mtp_embed.weight.set_value(paddle.ones_like(layer.mtp_embed.weight))
        ids = paddle.to_tensor([[1, 0, 2, 3, 4, 5]], dtype="int64")
        cu = paddle.to_tensor([0, 3, 6], dtype="int32")
        decoder = layer._prepare_erndata_magic_decoder_input(ids, cu)
        self.assertEqual(
            decoder.squeeze(-1).numpy().tolist(),
            [[0.0, 1.0, 0.0, 1.0, 1.0, 0.0]],
        )

    def test_fp32_residual_connection_promotes_decoder_input(self) -> None:
        layer, _ = self._make_layer(1, 0)
        layer.config.fp32_residual_connection = True
        object.__setattr__(
            layer, "mtp_embed", paddle.nn.Embedding(16, 1, dtype="float16")
        )
        ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6]], dtype="int64")
        cu = paddle.to_tensor([0, 3, 6], dtype="int32")
        decoder = layer._prepare_erndata_magic_decoder_input(ids, cu)
        self.assertEqual(decoder.dtype, paddle.float32)

    def test_magic_and_nonmagic_decoder_embeddings_match_k1_and_k2(self) -> None:
        """Identical tables yield identical packed decoder embeddings.

        This directly compares the magic path's local re-embedding with the
        ordinary path's precomputed float-embedding carrier.  It covers K=1 and
        both cumulative depths of K=2, including zero-filled document tails.
        """
        B, S = 1, 6
        ids = paddle.to_tensor([[1, 2, 3, 4, 5, 6]], dtype="int64")
        cu = paddle.to_tensor([0, 3, 6], dtype="int32")
        weight = paddle.arange(16, dtype="float32").reshape([16, 1])
        base = paddle.nn.functional.embedding(ids, weight)

        for K in (1, 2):
            ordinary_chunks = [base]
            rolled = base
            for _ in range(K):
                rolled, _ = roll_tensor(
                    rolled, shifts=-1, dims=1, cu_seqlens_q=cu
                )
                ordinary_chunks.append(rolled)

            ordinary_args = {
                "hidden_states": paddle.concat(ordinary_chunks),
                "cu_seqlens_q": cu,
                "context": None,
            }
            magic_args = {
                "hidden_states": base,
                "mtp_full_input_ids": ids,
                "cu_seqlens_q": cu,
                "context": None,
            }
            for depth in range(K):
                ordinary, ordinary_rec = self._make_layer(
                    K, depth, magic_send=False
                )
                magic, magic_rec = self._make_layer(K, depth)
                magic.mtp_embed.weight.set_value(ordinary.mtp_embed.weight)

                ordinary_args = ordinary._forward_megatron_style(ordinary_args)
                magic_args = magic._forward_megatron_style(magic_args)

                self.assertTrue(
                    paddle.equal_all(
                        magic_rec["decoder"], ordinary_rec["decoder"]
                    )
                )
                doc_tails = magic_rec["decoder"][:, [2, 5], :]
                self.assertTrue(
                    paddle.equal_all(
                        doc_tails, paddle.zeros_like(doc_tails)
                    )
                )

            self.assertIs(magic_args["cu_seqlens_q"], cu)
            self.assertNotIn("mtp_full_input_ids", magic_args)


class TestMaskFallbackWithoutInputIds(unittest.TestCase):
    """The fallback must not depend on the optional ``input_ids`` key.

    ``GPTEmbedding`` publishes ``input_ids`` as ``input_ids_for_moe_mask``,
    which stays None under plain ``use_erndata`` with
    ``expert_model_parallel_size == 1`` and
    ``gpt_model_use_experimental_version == False`` -- the key is then absent
    from ``dict_args``. The per-sample length must instead be recovered from
    the rank-local hidden_states shape and the parallel degrees.
    """

    def test_batch_flat_cu_without_input_ids(self) -> None:
        """B=2, L=4, batch-flat cu, no input_ids -> [2, 1, 4, 1] mask.

        Deriving L from a missing ``input_ids`` would fall back to per-sample
        semantics and emit a length-8 mask, which mismatches the length-4
        sequence attention actually sees.
        """
        K, B, L, h = 1, 2, 4, 2
        layer, rec = _make_mask_layer(K)
        # Carrier: K+1 slots concatenated along the batch axis, [ (K+1)*B, L, h ]
        carrier = paddle.zeros([(K + 1) * B, L, h], dtype="float32")
        cu = paddle.to_tensor([0, 2, 4, 7, 8], dtype="int32")
        layer._forward_megatron_style(
            {"hidden_states": carrier, "cu_seqlens_q": cu, "context": None}
        )
        self.assertEqual(rec["attn_mask_shape"], [B, 1, L, 1])
        self.assertEqual(_ends(rec["attn_mask"], 0), [2, 2, 4, 4])
        self.assertEqual(_ends(rec["attn_mask"], 1), [3, 3, 3, 4])

    def test_per_sample_cu_without_input_ids(self) -> None:
        """A per-sample cu keeps the shared length-L layout for both samples."""
        K, B, L, h = 1, 2, 4, 2
        layer, rec = _make_mask_layer(K)
        carrier = paddle.zeros([(K + 1) * B, L, h], dtype="float32")
        cu = paddle.to_tensor([0, 3, 4], dtype="int32")
        layer._forward_megatron_style(
            {"hidden_states": carrier, "cu_seqlens_q": cu, "context": None}
        )
        self.assertEqual(rec["attn_mask_shape"], [B, 1, L, 1])
        self.assertEqual(_ends(rec["attn_mask"], 0), [3, 3, 3, 4])
        self.assertEqual(_ends(rec["attn_mask"], 1), [3, 3, 3, 4])

    def test_sequence_parallel_scales_local_seq_back_up(self) -> None:
        """Under SP the seq axis holds L/TP; the mask must still cover L.

        Layout is seq-first, so the carrier concat is along the seq axis:
        [ (K+1)*L/TP, B, h ]. With TP=2 and local 4 the global L is 8, so a
        batch-flat cu spanning B*L=16 must be recognized. Forgetting the TP
        factor would make the span match neither reading and raise.
        """
        K, B, L, h, tp = 1, 2, 8, 2, 2
        local = L // tp
        layer, rec = _make_mask_layer(K, sequence_parallel=True, tp_size=tp)
        carrier = paddle.zeros([(K + 1) * local, B, h], dtype="float32")
        cu = paddle.to_tensor([0, 5, 8, 12, 16], dtype="int32")
        layer._forward_megatron_style(
            {"hidden_states": carrier, "cu_seqlens_q": cu, "context": None}
        )
        self.assertEqual(rec["attn_mask_shape"], [B, 1, L, 1])
        self.assertEqual(_ends(rec["attn_mask"], 0), [5, 5, 5, 5, 5, 8, 8, 8])
        self.assertEqual(_ends(rec["attn_mask"], 1), [4, 4, 4, 4, 8, 8, 8, 8])


if __name__ == "__main__":
    unittest.main()
