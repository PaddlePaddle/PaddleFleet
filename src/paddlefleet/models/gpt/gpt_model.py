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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddlefleet.spec_utils import LayerSpec

from paddle.distributed import fleet

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.lm_head import GPTLMHead
from paddlefleet.transformer.transformer_encoder import TransformerEncoder

from ...pipeline_parallel import (
    LayerDesc,
    SharedLayerDesc,
)

logger = logging.getLogger(__name__)


@dataclass
class GPTSublayersSpec:
    """p
    The dataclass for LayerSpecs of GPT sublayers_spec
    including embedding, n * transformer_layer, mtp, lm_head.
    """

    embedding: LayerSpec | None = None
    head_empty_layers: list[LayerSpec] | None = None
    transformer_layers: list[LayerSpec] | None = None
    tail_empty_layers: list[LayerSpec] | None = None
    mtp: list[LayerSpec] | None = None
    layer_norm: LayerSpec | None = None
    lm_head: LayerSpec | None = None


class GPTModel(TransformerEncoder):
    """GPT Transformer language model.

    Args:
        gpt_layer_desc:
    """

    def __init__(
        self,
        sublayers_spec: GPTSublayersSpec,
        **kwargs,
    ) -> None:
        self.config = kwargs["config"]
        self.modal = kwargs.pop("modal", None)
        tie_word_embeddings = (
            kwargs["tie_word_embeddings"]
            and self.config.pipeline_model_parallel_size > 1
        )
        skip_weight_param_allocation = (
            self.config.tie_word_embeddings
            and self.config.pipeline_model_parallel_size == 1
        )
        self._pipeline_name_mapping = None
        self._pp_to_single_mapping = None
        self._sequential_layers = self.get_layer_desc_list(
            sublayers_spec,
            tie_word_embeddings,
        )
        self.layers = self.get_sequential_layers()
        del kwargs["tie_word_embeddings"]
        del kwargs["config"]

        topology = (
            None
            if self.config.pipeline_model_parallel_size == 1
            else fleet.get_hybrid_communicate_group().topology()
        )

        super(TransformerEncoder, self).__init__(
            layers=self.layers,
            topology=topology,
            num_virtual_pipeline_stages=self.config.virtual_pipeline_model_parallel_size,
            **kwargs,
        )

        if skip_weight_param_allocation:
            shared_embed_weight = None
            for layer in self.run_function:
                if isinstance(layer, GPTEmbedding):
                    shared_embed_weight = layer.embedding_weight
                if isinstance(layer, GPTLMHead):
                    layer.weight = shared_embed_weight

    def get_layer_desc_list(self, spec, tie_word_embeddings):
        layers = []
        if self.modal:
            name_prefix = f"model.{self.modal}"
        else:
            name_prefix = "model"
        if tie_word_embeddings:
            self.add_sequential_layer(
                layers,
                SharedLayerDesc(
                    "embed",
                    spec.embedding,
                    shared_weight_attr="embedding_weight",
                ),
                name_prefix,
            )
        else:
            self.add_sequential_layer(
                layers, LayerDesc(spec.embedding), name_prefix
            )
        i = 0
        for head_empty_layer in spec.head_empty_layers:
            self.add_sequential_layer(
                layers, LayerDesc(head_empty_layer), f"{name_prefix}.layers.{i}"
            )
            i += 1
        for transformer_layer_spec in spec.transformer_layers:
            self.add_sequential_layer(
                layers,
                LayerDesc(transformer_layer_spec),
                f"{name_prefix}.layers.{i}",
            )
            i += 1

        # Always place layer_norm after transformer_layers and before tail_empty_layers/MTP,
        # so that the model structure is consistent regardless of whether MTP is enabled.
        self.add_sequential_layer(
            layers, LayerDesc(spec.layer_norm), name_prefix
        )

        if spec.mtp:
            for mtp_spec in spec.mtp:
                self.add_sequential_layer(
                    layers, LayerDesc(mtp_spec), f"{name_prefix}.layers.{i}"
                )
                i += 1
        for tail_empty_layer in spec.tail_empty_layers:
            self.add_sequential_layer(
                layers, LayerDesc(tail_empty_layer), f"{name_prefix}.layers.{i}"
            )
            i += 1

        if tie_word_embeddings:
            self.add_sequential_layer(
                layers,
                SharedLayerDesc(
                    "embed",
                    spec.lm_head,
                    shared_weight_attr="embedding_weight",
                ),
                f"{name_prefix}.shared_head",
            )
        else:
            self.add_sequential_layer(
                layers, LayerDesc(spec.lm_head), f"{name_prefix}.lm_head"
            )

        return layers
