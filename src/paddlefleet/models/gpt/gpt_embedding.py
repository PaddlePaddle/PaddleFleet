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
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from paddlefleet.spec_utils import LayerSpec, build_layer
from paddlefleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddle import Tensor

    from paddlefleet.packed_seq_params import PackedSeqParams
    from paddlefleet.transformer.transformer_config import TransformerConfig


@dataclass
class GPTEmbeddingSpec:
    language_embedding: LayerSpec
    rope_embedding: LayerSpec | None


class GPTEmbedding(FleetLayer):
    def __init__(
        self,
        sublayers_spec: GPTEmbeddingSpec,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "none"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
    ):
        super().__init__(config)
        self.embedding = build_layer(
            sublayers_spec.language_embedding,
            config=config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
        )
        self.rotary_pos_emb = None
        if sublayers_spec.rope_embedding is not None:
            self.rotary_pos_emb = build_layer(
                sublayers_spec.rope_embedding,
                head_dim=config.head_dim,
                rotary_percent=rotary_percent,
                rotary_interleaved=config.rotary_interleaved,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
            )

    @property
    def embedding_weight(self):
        return self.embedding.embedding_weight

    def forward(
        self,
        dict_args: dict,
        decoder_input: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        input_ids = dict_args["input_ids"]
        position_ids = dict_args.get("position_ids", None)
        attention_mask = dict_args.get("attention_mask", None)
        attn_mask_startend_row_indices = dict_args.get(
            "attn_mask_startend_row_indices", None
        )

        if decoder_input is None:
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None

        if self.rotary_pos_emb is not None:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
            )
            if self.config.sequence_parallel:
                rotary_pos_emb = rotary_pos_emb.transpose(
                    [1, 0, 2, 3]
                ).contiguous()

        preproc_output = {
            "hidden_states": decoder_input,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "rotary_pos_emb": rotary_pos_emb,
        }

        for key in list(preproc_output.keys()):
            if preproc_output[key] is None:
                preproc_output.pop(key)

        return preproc_output
