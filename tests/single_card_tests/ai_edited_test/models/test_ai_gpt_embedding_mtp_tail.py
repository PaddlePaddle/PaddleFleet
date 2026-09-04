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
"""Regression tests for the MTP carrier-tail convention in ``GPTEmbedding.forward``.

The MTP branch consumes an offset slice ``ids[d + 1 : d + 1 + S]`` of the carrier
sequence, which the collate pads to ``max_seq_len + num_nextn_predict_layers``. That
makes the positions vacated by the shift carry the REAL trailing carrier tokens (the
pad token). The reference implementation instead rolls by -1 and zero-fills, so at
depth ``d`` its last ``d + 1`` MTP positions embed token id 0.

Under IEEE+UAC the carrier tail is zeroed so both conventions agree. FLAG+UAC
alone leaves the structure carrier. These tests pin that behaviour, its gate, and
the fact that the main path is untouched.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

PAD_TOKEN_ID = 154820


def _make_embedding(*, use_accuracy_compatible, num_nextn_predict_layers):
    """Build a GPTEmbedding shell whose forward can run without a real model."""
    emb = GPTEmbedding.__new__(GPTEmbedding)
    emb.__dict__.setdefault("_parameters", {})
    emb.__dict__.setdefault("_buffers", {})
    emb.__dict__.setdefault("_sub_layers", {})
    emb.__dict__.setdefault("_loaddict_holder", {})
    emb.__dict__.setdefault("_non_persistable_buffers", set())
    emb.__dict__.setdefault("_non_persistable_buffer_names_set", set())
    emb.config = MagicMock()
    emb.config.sequence_parallel = False
    emb.config.multimodal_embedding = False
    emb.config.expert_model_parallel_size = 1
    emb.config.tensor_model_parallel_size = 2
    emb.config.num_nextn_predict_layers = num_nextn_predict_layers
    emb.config.mtp_load_weight_only = False
    emb.config.apply_rope_fusion = False
    emb.config.experimental_dataflow = False
    emb.config.gpt_model_use_experimental_version = False
    emb.config.enable_mtp_magic_send = False
    emb.config.use_accuracy_compatible = use_accuracy_compatible
    emb.config.pad_token_id = PAD_TOKEN_ID
    emb.multimodal_embedding = False
    emb.position_embedding_type = "none"
    emb.rotary_pos_emb = None
    emb.swa_rotary_pos_emb = None
    emb.mrope_section = None
    emb.sequence_parallel = False

    seen = {}

    def _embed(input_ids=None, position_ids=None):
        seen["input_ids"] = input_ids.clone()
        return paddle.zeros([input_ids.shape[0], input_ids.shape[1], 4])

    emb.embedding = _embed
    return emb, seen


def _carrier(seq_len, tail):
    """A carrier row of distinct non-pad ids followed by ``tail`` pad tokens."""
    body = list(range(10, 10 + seq_len - tail))
    return paddle.to_tensor([body + [PAD_TOKEN_ID] * tail], dtype="int64")


class TestGPTEmbeddingMTPCarrierTail(unittest.TestCase):
    """The tail zeroing must be exact, gated, and confined to the tail."""

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_tail_is_zeroed_under_accuracy_compatible(self):
        emb, seen = _make_embedding(
            use_accuracy_compatible=True, num_nextn_predict_layers=1
        )
        input_ids = _carrier(seq_len=8, tail=3)
        emb.forward(dict_args={"input_ids": input_ids})

        embedded = seen["input_ids"].numpy().tolist()[0]
        self.assertEqual(
            embedded[-1], 0, "the vacated MTP position must embed id 0"
        )
        self.assertEqual(
            embedded[:-1],
            input_ids.numpy().tolist()[0][:-1],
            "only the last num_nextn_predict_layers ids may change",
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_tail_length_follows_depth(self):
        emb, seen = _make_embedding(
            use_accuracy_compatible=True, num_nextn_predict_layers=3
        )
        input_ids = _carrier(seq_len=10, tail=4)
        emb.forward(dict_args={"input_ids": input_ids})

        embedded = seen["input_ids"].numpy().tolist()[0]
        self.assertEqual(
            embedded[-3:], [0, 0, 0], "depth d must leave d trailing zero ids"
        )
        self.assertNotEqual(
            embedded[-4], 0, "the zeroing must not reach further back"
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_disabled_without_accuracy_compatible(self):
        emb, seen = _make_embedding(
            use_accuracy_compatible=False, num_nextn_predict_layers=1
        )
        input_ids = _carrier(seq_len=8, tail=3)
        emb.forward(dict_args={"input_ids": input_ids})

        self.assertEqual(
            seen["input_ids"].numpy().tolist(),
            input_ids.numpy().tolist(),
            "without the alignment switch the carrier must be untouched",
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_disabled_without_mtp(self):
        emb, seen = _make_embedding(
            use_accuracy_compatible=True, num_nextn_predict_layers=0
        )
        input_ids = _carrier(seq_len=8, tail=3)
        emb.forward(dict_args={"input_ids": input_ids})

        self.assertEqual(
            seen["input_ids"].numpy().tolist(),
            input_ids.numpy().tolist(),
            "with MTP off there is no shifted slice to align",
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_dict_args_is_updated_for_downstream_consumers(self):
        emb, _ = _make_embedding(
            use_accuracy_compatible=True, num_nextn_predict_layers=1
        )
        input_ids = _carrier(seq_len=8, tail=3)
        dict_args = {"input_ids": input_ids}
        emb.forward(dict_args=dict_args)

        self.assertEqual(
            dict_args["input_ids"].numpy().tolist()[0][-1],
            0,
            "downstream MTP consumers read input_ids back out of dict_args",
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "0"})
    def test_flag_uac_without_ieee_leaves_carrier(self):
        emb, seen = _make_embedding(
            use_accuracy_compatible=True, num_nextn_predict_layers=1
        )
        input_ids = _carrier(seq_len=8, tail=3)
        emb.forward(dict_args={"input_ids": input_ids})

        self.assertEqual(
            seen["input_ids"].numpy().tolist(),
            input_ids.numpy().tolist(),
            "FLAG+UAC without IEEE must keep the structure carrier",
        )

    @patch.dict(os.environ, {"MODEL_REPRO_IEEE_KERNEL": "1"})
    def test_shorter_than_depth_is_left_alone(self):
        emb, seen = _make_embedding(
            use_accuracy_compatible=True, num_nextn_predict_layers=4
        )
        input_ids = paddle.to_tensor([[11, 12, 13, 14]], dtype="int64")
        # Carrier length equals depth, so the IEEE tail-zero gate does not
        # fire. The mock embedding still runs; this only pins that the
        # whole carrier is not blanked.
        try:
            emb.forward(dict_args={"input_ids": input_ids})
        except ValueError:
            pass
        self.assertEqual(
            seen["input_ids"].numpy().tolist(),
            input_ids.numpy().tolist(),
            "equal-length carriers must not be blanked by the IEEE tail gate",
        )

        self.assertEqual(
            seen["input_ids"].numpy().tolist(),
            input_ids.numpy().tolist(),
            "a carrier no longer than the depth must not be blanked entirely",
        )


if __name__ == "__main__":
    unittest.main()
