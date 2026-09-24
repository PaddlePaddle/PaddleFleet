# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Accuracy-compatible MTP paths of ``GPTEmbedding``.

``GPTEmbedding.forward`` is a plain function, so it is driven with a
namespace instance: a recording lookup stands in for
``LanguageModelEmbedding`` and the collectives are replaced by identity
stubs, which keeps everything single-card while still running the real
control flow. The tests pin the second (detached-tail / per-depth) lookups
the accuracy-compatible mode adds and the MTP payload it hands to the
pipeline.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import LayerSpec

import paddlefleet.parallel_state as ps
from paddlefleet.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from paddlefleet.models.gpt import gpt_embedding as ge
from paddlefleet.transformer.transformer_config import TransformerConfig

VOCAB = 10
HIDDEN = 4
BATCH = 2
SEQ = 6
MTP_DEPTH = 2
# The two trailing columns are the MTP carrier; accuracy-compatible mode
# zero-fills them before the main lookup.
RAW_IDS = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 1, 2, 3]]
ZEROED_IDS = np.array([[1, 2, 3, 4, 0, 0], [7, 8, 9, 1, 0, 0]])
TABLE = np.arange(VOCAB * HIDDEN, dtype="float32").reshape(VOCAB, HIDDEN)


def _init_fleet():
    if not ps.have_global_memory_buffer():
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        fleet.init(is_collective=True, strategy=strategy)
        ps.initialize_model_parallel(fleet.get_hybrid_communicate_group())


def _expected_lookups():
    """Reference embeddings for the main, carrier and per-depth lookups."""
    main_ids = ZEROED_IDS[:, :-MTP_DEPTH]
    tail_ids = ZEROED_IDS[:, -MTP_DEPTH:]
    main = TABLE[main_ids]
    extra = TABLE[tail_ids]
    shifted = [
        np.concatenate([main[:, depth + 1 :], extra[:, : depth + 1]], axis=1)
        for depth in range(MTP_DEPTH)
    ]
    return main_ids, tail_ids, main, shifted


class _RecordingEmbedding:
    """Deterministic stand-in for ``LanguageModelEmbedding.__call__``."""

    def __init__(self, table):
        self.table = table
        self.tp_group = None
        self.calls = []

    def __call__(self, input_ids=None, position_ids=None):
        self.calls.append(
            (
                input_ids.numpy().copy(),
                None if position_ids is None else position_ids.numpy().copy(),
            )
        )
        return F.embedding(input_ids, self.table)


def _config(**overrides):
    config = SimpleNamespace(
        gpt_model_use_experimental_version=False,
        use_accuracy_compatible=True,
        num_nextn_predict_layers=MTP_DEPTH,
        mtp_load_weight_only=False,
        use_erndata=False,
        enable_mtp_magic_send=False,
        experimental_dataflow=False,
        cp_balance_mode="dualchunk",
        expert_model_parallel_size=1,
        tensor_model_parallel_size=1,
        separate_mtp_input=True,
        sequence_parallel=False,
        apply_rope_fusion=False,
        clone_scatter_output_in_embedding=False,
        multimodal_embedding=False,
        pad_token_id=0,
    )
    config.__dict__.update(overrides)
    return config


def _instance(config, embedding, sequence_parallel=False):
    return SimpleNamespace(
        config=config,
        embedding=embedding,
        multimodal_embedding=False,
        sequence_parallel=sequence_parallel,
        rotary_pos_emb=None,
        swa_rotary_pos_emb=None,
        position_embedding_type="none",
        mrope_section=None,
        has_kda_layer=False,
        training=True,
    )


def _table():
    table = paddle.to_tensor(TABLE)
    table.stop_gradient = False
    return table


class _ForwardHarness(unittest.TestCase):
    """Runs ``GPTEmbedding.forward`` with single-card collective stubs."""

    def _run_forward(
        self, config, dict_args, decoder_input=None, cp_size=1, **kwargs
    ):
        embedding = _RecordingEmbedding(_table())
        instance = _instance(config, embedding, **kwargs)
        cp_scatter = Mock(side_effect=lambda value, axis=None, mode=None: value)
        sp_scatter = Mock(side_effect=lambda value: value)
        with (
            patch.object(
                ge,
                "inspect_tensor",
                side_effect=lambda name, layer, value: value,
            ),
            patch.object(ge, "get_pg_size", return_value=1),
            patch.object(
                ge, "get_context_parallel_world_size", return_value=cp_size
            ),
            patch.object(
                ge, "use_dsv4_accuracy_compatible", return_value=False
            ),
            patch.object(
                ge,
                "ContextParallelScatterOp",
                SimpleNamespace(apply=cp_scatter),
            ),
            patch.object(ge, "ScatterOp", SimpleNamespace(apply=sp_scatter)),
        ):
            output = ge.GPTEmbedding.forward(
                instance, dict_args, decoder_input=decoder_input
            )
        return output, embedding, cp_scatter, sp_scatter


class AccuracyCompatibleMtpLookupTest(_ForwardHarness):
    """The carrier tail is zero-filled and looked up a second time."""

    def _dict_args(self):
        return {
            "input_ids": paddle.to_tensor(RAW_IDS, dtype="int64"),
            "position_ids": paddle.to_tensor(
                [list(range(SEQ))] * BATCH, dtype="int64"
            ),
        }

    def test_second_lookup_ids_and_truncated_positions(self):
        dict_args = self._dict_args()
        main_ids, tail_ids, _, _ = _expected_lookups()

        _, embedding, _, _ = self._run_forward(_config(), dict_args)

        # The carrier tail is zeroed in place for every later consumer.
        np.testing.assert_array_equal(
            dict_args["input_ids"].numpy(), ZEROED_IDS
        )
        self.assertEqual(len(embedding.calls), 2 + MTP_DEPTH)
        # Main lookup: leading L-K ids with position_ids cut to match.
        np.testing.assert_array_equal(embedding.calls[0][0], main_ids)
        np.testing.assert_array_equal(
            embedding.calls[0][1],
            np.array([list(range(SEQ - MTP_DEPTH))] * BATCH),
        )
        # Carrier lookup: the zeroed tail, never any position id.
        np.testing.assert_array_equal(embedding.calls[1][0], tail_ids)
        self.assertIsNone(embedding.calls[1][1])
        # Per-depth lookups roll the semantic ids and zero-fill the rest.
        for depth in range(MTP_DEPTH):
            expected = np.concatenate(
                [
                    main_ids[:, depth + 1 :],
                    np.zeros_like(main_ids[:, : depth + 1]),
                ],
                axis=1,
            )
            with self.subTest(depth=depth):
                np.testing.assert_array_equal(
                    embedding.calls[2 + depth][0], expected
                )
                self.assertIsNone(embedding.calls[2 + depth][1])

    def test_separate_mtp_input_payload(self):
        _, _, main, shifted = _expected_lookups()

        output, _, cp_scatter, sp_scatter = self._run_forward(
            _config(separate_mtp_input=True), self._dict_args()
        )

        cp_scatter.assert_not_called()
        sp_scatter.assert_not_called()
        # Backbone keeps only the main chunk ...
        np.testing.assert_array_equal(output["hidden_states"].numpy(), main)
        # ... and the shifted chunks travel on their own key.
        stacked = output["mtp_decoder_inputs"]
        self.assertEqual(
            list(stacked.shape), [MTP_DEPTH, BATCH, SEQ - MTP_DEPTH, HIDDEN]
        )
        np.testing.assert_array_equal(stacked.numpy(), np.stack(shifted))
        # The re-attached lookup keeps the payload differentiable.
        self.assertFalse(stacked.stop_gradient)
        self.assertNotIn("labels", output)

    def test_concat_mtp_input_payload(self):
        _, _, main, shifted = _expected_lookups()

        output, _, _, _ = self._run_forward(
            _config(separate_mtp_input=False), self._dict_args()
        )

        self.assertNotIn("mtp_decoder_inputs", output)
        np.testing.assert_array_equal(
            output["hidden_states"].numpy(),
            np.concatenate([main, *shifted], axis=0),
        )

    def test_context_and_sequence_parallel_scatters_run(self):
        _, _, main, shifted = _expected_lookups()

        output, embedding, cp_scatter, sp_scatter = self._run_forward(
            _config(
                separate_mtp_input=False,
                experimental_dataflow=True,
                sequence_parallel=True,
            ),
            self._dict_args(),
            cp_size=2,
            sequence_parallel=True,
        )

        # Main chunk once, then twice per depth: the shifted slice and the
        # re-attached per-depth lookup are scattered with the same layout.
        self.assertEqual(cp_scatter.call_count, 1 + 2 * MTP_DEPTH)
        self.assertEqual(sp_scatter.call_count, 1 + 2 * MTP_DEPTH)
        self.assertEqual(len(embedding.calls), 2 + MTP_DEPTH)
        # Chunks leave in [s, b, h] once sequence parallel is on.
        expected = np.concatenate(
            [chunk.transpose(1, 0, 2) for chunk in (main, *shifted)], axis=0
        )
        self.assertEqual(
            list(output["hidden_states"].shape),
            [(MTP_DEPTH + 1) * (SEQ - MTP_DEPTH), BATCH, HIDDEN],
        )
        np.testing.assert_array_equal(output["hidden_states"].numpy(), expected)


class MtpAttentionMaskPassthroughTest(_ForwardHarness):
    """Per-depth MTP masks ride along with the pipeline payload."""

    def _decoder_input(self):
        value = paddle.to_tensor(
            np.arange(BATCH * SEQ * HIDDEN, dtype="float32").reshape(
                BATCH, SEQ, HIDDEN
            )
        )
        value.stop_gradient = False
        return value

    def test_dense_mask_is_forwarded(self):
        decoder_input = self._decoder_input()
        dense = paddle.to_tensor(
            np.tril(np.ones((MTP_DEPTH, SEQ), dtype="int32"))
        )
        hidden_mask = paddle.to_tensor(
            np.ones((MTP_DEPTH, BATCH, SEQ), dtype="int32")
        )
        dict_args = {
            "input_ids": None,
            "mtp_attn_mask": dense,
            "mtp_hidden_inputs_mask_all": hidden_mask,
        }

        output, embedding, _, _ = self._run_forward(
            _config(), dict_args, decoder_input=decoder_input
        )

        # An external decoder_input skips every embedding lookup.
        self.assertEqual(embedding.calls, [])
        np.testing.assert_array_equal(
            output["hidden_states"].numpy(), decoder_input.numpy()
        )
        self.assertNotIn("mtp_startend_row_indices_all", output)
        np.testing.assert_array_equal(
            output["mtp_attn_mask"].numpy(), dense.numpy()
        )
        np.testing.assert_array_equal(
            output["mtp_hidden_inputs_mask_all"].numpy(), hidden_mask.numpy()
        )
        self.assertNotIn("mtp_decoder_inputs", output)

    def test_compressed_mask_is_forwarded(self):
        compressed = paddle.to_tensor(
            np.ones((MTP_DEPTH, BATCH, SEQ, 1), dtype="int32")
        )
        hidden_mask = paddle.to_tensor(
            np.ones((MTP_DEPTH, BATCH, SEQ), dtype="int32")
        )
        dict_args = {
            "input_ids": None,
            "mtp_startend_row_indices_all": compressed,
            "mtp_hidden_inputs_mask_all": hidden_mask,
        }

        output, _, _, _ = self._run_forward(
            _config(), dict_args, decoder_input=self._decoder_input()
        )

        np.testing.assert_array_equal(
            output["mtp_startend_row_indices_all"].numpy(),
            compressed.numpy(),
        )
        self.assertNotIn("mtp_attn_mask", output)

    def test_mask_combinations_are_rejected(self):
        ones = paddle.to_tensor(np.ones((MTP_DEPTH, BATCH, SEQ), "int32"))
        rejected = {
            "dense and compressed together": {
                "mtp_attn_mask": ones,
                "mtp_startend_row_indices_all": ones,
                "mtp_hidden_inputs_mask_all": ones,
            },
            "mask without hidden input mask": {"mtp_attn_mask": ones},
            "hidden input mask without mask": {
                "mtp_hidden_inputs_mask_all": ones
            },
        }
        for reason, extra in rejected.items():
            dict_args = {"input_ids": None, **extra}
            with (
                self.subTest(reason=reason),
                self.assertRaises(AssertionError),
            ):
                self._run_forward(
                    _config(),
                    dict_args,
                    decoder_input=self._decoder_input(),
                )


class AccuracyCompatibleMainGradClaimTest(unittest.TestCase):
    """UAC claims ``main_grad`` on the embedding weight at build time."""

    def setUp(self):
        _init_fleet()

    @staticmethod
    def _build(use_accuracy_compatible):
        config = TransformerConfig(
            num_hidden_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=True,
            use_accuracy_compatible=use_accuracy_compatible,
        )
        spec = ge.GPTEmbeddingSpec(
            language_embedding=LayerSpec(layer=LanguageModelEmbedding),
            rope_embedding=None,
        )
        return ge.GPTEmbedding(
            spec,
            config,
            vocab_size=32,
            max_sequence_length=8,
            position_embedding_type="none",
        )

    def test_main_grad_slot_is_claimed_under_accuracy_compatible(self):
        embedding = self._build(True)
        weight = embedding.embedding.embed_tokens.weight
        # Claimed but empty: MixPrecision must skip this parameter and the
        # indexing backward fills the buffer later.
        self.assertTrue(hasattr(weight, "main_grad"))
        self.assertIsNone(weight.main_grad)

    def test_main_grad_slot_is_absent_by_default(self):
        embedding = self._build(False)
        weight = embedding.embedding.embed_tokens.weight
        self.assertFalse(hasattr(weight, "main_grad"))


if __name__ == "__main__":
    unittest.main()
