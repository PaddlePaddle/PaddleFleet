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

"""MTP mask passthrough of ``GPTEmbedding``.

``GPTEmbedding.forward`` is a plain function, so it is driven with a
namespace instance: a recording lookup stands in for
``LanguageModelEmbedding`` and the collectives are replaced by identity
stubs, which keeps everything single-card while still running the real
control flow. The tests pin the per-depth MTP masks it hands to the
pipeline.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.models.gpt import gpt_embedding as ge

VOCAB = 10
HIDDEN = 4
BATCH = 2
SEQ = 6
MTP_DEPTH = 2
TABLE = np.arange(VOCAB * HIDDEN, dtype="float32").reshape(VOCAB, HIDDEN)


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


if __name__ == "__main__":
    unittest.main()
