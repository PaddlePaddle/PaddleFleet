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

"""Kimi-K3 vision tower (MoonViT3d) assembled as a PaddleFleet pipeline."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paddle.distributed.fleet.meta_parallel import LayerDesc, LayerSpec
from paddle.distributed.fleet.utils import recompute

from ...transformer.transformer_encoder import TransformerEncoder
from ...transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)

if TYPE_CHECKING:
    import paddle

    from ...packed_seq_params import PackedSeqParams
    from ...process_groups_config import ProcessGroupCollection
    from ...transformer.transformer_config import TransformerConfig


@dataclass
class KimiK3VisionSublayersSpec:
    """LayerSpecs of the Kimi-K3 vision tower stages: embedding,
    n * transformer_layer, final_layernorm, sd2_tpool merge, mm_projector.
    """

    embedding: LayerSpec = None
    head_empty_layers: list[LayerSpec] = None
    transformer_layers: list[LayerSpec] = None
    tail_empty_layers: list[LayerSpec] = None
    final_layernorm: LayerSpec = None
    sdtpool_merger: LayerSpec = None
    merger: LayerSpec = None


class KimiK3VisionModel(TransformerEncoder):
    """MoonViT3d + multimodal projector, pipeline-parallel capable."""

    def get_layer_desc_list(self, spec: KimiK3VisionSublayersSpec):
        layers = []
        if self.modal:
            name_prefix = f"model.{self.modal}"
        else:
            name_prefix = "model"

        self.add_sequential_layer(
            layers, LayerDesc(spec.embedding), f"{name_prefix}.patch_embed"
        )
        self.get_encoder_layer_desc_list(layers, spec, name_prefix)
        self.add_sequential_layer(
            layers,
            LayerDesc(spec.final_layernorm),
            f"{name_prefix}.final_layernorm",
        )
        # no parameter
        self.add_sequential_layer(
            layers,
            LayerDesc(spec.sdtpool_merger),
            f"{name_prefix}.sdtpool_merger",
        )
        self.add_sequential_layer(
            layers, LayerDesc(spec.merger), f"{name_prefix}.mm_projector"
        )

        return layers


class KimiK3VisionTransformerLayer(TransformerLayer):
    """MoonViT encoder layer: dict in / dict out so that ``grid_thws`` and the
    shared 2D RoPE ``rope_freqs_cis`` survive the pipeline.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: TransformerLayerSublayersSpec,
        layer_number: int = 1,
        hidden_dropout_prob: float | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        modal: str | None = None,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            hidden_dropout_prob=hidden_dropout_prob,
            pg_collection=pg_collection,
        )
        self.modal = modal

    def forward(self, dict_args: dict):
        dict_args.pop("dynamic_inference_decode_only", None)
        dict_args.pop("position_ids", None)

        if self.full_recompute:
            attn_mask_startend_row_indices = dict_args.get(
                "attn_mask_startend_row_indices", None
            )
            rope_freqs_cis = dict_args.get("rope_freqs_cis", None)
            outputs = recompute(
                self._forward_impl,
                hidden_states=dict_args["hidden_states"],
                attention_mask=dict_args.get("attention_mask", None),
                # Clone is necessary!
                attn_mask_startend_row_indices=attn_mask_startend_row_indices.clone()
                if attn_mask_startend_row_indices is not None
                else None,
                context=dict_args.get("context", None),
                context_mask=dict_args.get("context_mask", None),
                rope_freqs_cis=rope_freqs_cis.clone()
                if rope_freqs_cis is not None
                else None,  # Clone is necessary!
                attention_bias=dict_args.get("attention_bias", None),
                packed_seq_params=dict_args.get("packed_seq_params", None),
            )
        else:
            forward_args = {
                key: value
                for key, value in dict_args.items()
                if key
                in (
                    "hidden_states",
                    "attention_mask",
                    "attn_mask_startend_row_indices",
                    "context",
                    "context_mask",
                    "rope_freqs_cis",
                    "attention_bias",
                    "packed_seq_params",
                )
            }
            outputs = self._forward_impl(**forward_args)

        if len(outputs) == 3:
            output, context = outputs[0], outputs[1]
        else:
            output, context = outputs, None

        rst = OrderedDict(dict_args)
        rst["hidden_states"] = output
        if context is not None:
            rst["context"] = context
        return rst

    def _forward_impl(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor = None,
        attn_mask_startend_row_indices: paddle.Tensor = None,
        context: paddle.Tensor = None,
        context_mask: paddle.Tensor = None,
        rope_freqs_cis: paddle.Tensor = None,
        attention_bias: paddle.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        # seq, hidden_size -> batch, seq, hidden_size
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)

        hidden_states, context = self._forward_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_mask_startend_row_indices=attn_mask_startend_row_indices,
            context=context,
            context_mask=context_mask,
            rope_freqs_cis=rope_freqs_cis,
            attention_bias=attention_bias,
            in_recompute=self.full_recompute,
        )
        hidden_states = self._forward_mlp(hidden_states)

        if context is not None:
            return hidden_states, context
        return hidden_states
