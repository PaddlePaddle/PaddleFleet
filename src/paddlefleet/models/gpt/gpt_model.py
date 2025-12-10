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
from typing import TYPE_CHECKING

from paddlefleet.pipeline_parallel import (
    LayerDesc,
    PipelineLayer,
    SharedLayerDesc,
)

if TYPE_CHECKING:
    from paddlefleet.spec_utils import LayerSpec

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.models.gpt.lm_head import GPTLMHead


@dataclass
class GPTSublayersSpec:
    """
    The dataclass for LayerSpecs of GPT sublayers_spec
    including embedding, n * transformer_layer, mtp, lm_head.
    """

    embedding: LayerSpec | None = None
    transformer_layers: list[LayerSpec] | None = None
    layer_norm: LayerSpec | None = None
    mtp: list[LayerSpec] | None = None
    lm_head: LayerSpec | None = None


class GPTModel(PipelineLayer):
    """GPT Transformer language model.

    Args:
        gpt_layer_desc:
    """

    def __init__(
        self,
        sublayers_spec: GPTSublayersSpec,
        **kwargs,
    ) -> None:
        config = kwargs["config"]
        share_embeddings_and_output_weights = (
            kwargs["share_embeddings_and_output_weights"]
            and config.pipeline_model_parallel_size > 1
        )
        skip_weight_param_allocation = (
            config.share_embeddings_and_output_weights
            and config.pipeline_model_parallel_size == 1
        )
        self.layers = GPTModel.get_layer_desc_list(
            sublayers_spec,
            share_embeddings_and_output_weights,
        )
        del kwargs["share_embeddings_and_output_weights"]
        del kwargs["config"]

        super().__init__(layers=self.layers, **kwargs)

        if skip_weight_param_allocation:
            shared_embed_weight = None
            for layer in self.run_function:
                if isinstance(layer, GPTEmbedding):
                    shared_embed_weight = layer.embedding_weight
                if isinstance(layer, GPTLMHead):
                    layer.weight = shared_embed_weight

    @staticmethod
    def get_layer_desc_list(spec, share_embeddings_and_output_weights):
        if share_embeddings_and_output_weights:
            layers = [
                SharedLayerDesc(
                    "embed",
                    spec.embedding,
                    shared_weight_attr="embedding_weight",
                )
            ]
        else:
            layers = [LayerDesc(spec.embedding)]

        for transformer_layer_spec in spec.transformer_layers:
            layers.append(LayerDesc(transformer_layer_spec))

        layers.append(LayerDesc(spec.layer_norm))

        if spec.mtp is not None:
            for mtp_spec in spec.mtp:
                layers.append(LayerDesc(mtp_spec))

        if share_embeddings_and_output_weights:
            layers.append(
                SharedLayerDesc(
                    "embed",
                    spec.lm_head,
                    shared_weight_attr="embedding_weight",
                )
            )
        else:
            layers.append(LayerDesc(spec.lm_head))

        return layers

    def get_hardware_flops(self):
        return 989e3
