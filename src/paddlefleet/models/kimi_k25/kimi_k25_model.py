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
from collections import OrderedDict
from dataclasses import dataclass

import paddle
from paddle.distributed.fleet.utils import recompute

from ...packed_seq_params import PackedSeqParams
from ...pipeline_parallel import LayerDesc
from ...process_groups_config import ProcessGroupCollection
from ...spec_utils import LayerSpec
from ...transformer.transformer_config import TransformerConfig
from ...transformer.transformer_encoder import TransformerEncoder
from ...transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)


@dataclass
class KimiK25VisionSublayersSpec:
    """
    The dataclass for LayerSpecs of Qwen3-VL vision model sublayers_spec,
    including embedding, n * transformer_layer, patch_merger, deepstack_merger.
    """

    embedding: LayerSpec = None
    head_empty_layers: list[LayerSpec] = None
    transformer_layers: list[LayerSpec] = None
    tail_empty_layers: list[LayerSpec] = None
    sdtpool_merger: LayerSpec = None
    merger: LayerSpec = None


class KimiK25VisionModel(TransformerEncoder):
    def get_layer_desc_list(self, spec: KimiK25VisionSublayersSpec):
        layers = []
        if self.modal:
            name_prefix = f"model.{self.modal}"
        else:
            name_prefix = "model"

        self.add_sequential_layer(
            layers, LayerDesc(spec.embedding), f"{name_prefix}.patch_embed"
        )
        self.get_encoder_layer_desc_list(layers, spec, name_prefix)
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


class KimiK25VisionTransformerLayer(TransformerLayer):
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

    def forward(
        self,
        dict_args: dict,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        dict_args.pop("dynamic_inference_decode_only", None)
        dict_args.pop("position_ids", None)

        if self.full_recompute:
            hidden_states = dict_args["hidden_states"]
            attention_mask = dict_args.get("attention_mask", None)
            attn_mask_startend_row_indices = dict_args.get(
                "attn_mask_startend_row_indices", None
            )
            context = dict_args.get("context", None)
            context_mask = dict_args.get("context_mask", None)
            rotary_pos_emb = dict_args.get("rotary_pos_emb", None)
            rotary_pos_cos = dict_args.get("rotary_pos_cos", None)
            rotary_pos_sin = dict_args.get("rotary_pos_sin", None)
            attention_bias = dict_args.get("attention_bias", None)
            packed_seq_params = dict_args.get("packed_seq_params", None)

            assert (rotary_pos_sin is None) == (rotary_pos_cos is None)

            if rotary_pos_cos is not None and rotary_pos_sin is not None:
                rotary_pos_cos = rotary_pos_cos.clone()
                rotary_pos_sin = rotary_pos_sin.clone()
                if self.config.apply_rope_fusion:
                    rotary_pos_cos = rotary_pos_cos[0, ...]
                    rotary_pos_sin = rotary_pos_sin[0, ...]
                    if rotary_pos_cos.ndim == 2:
                        rotary_pos_cos = rotary_pos_cos.reshape(
                            [
                                1,
                                rotary_pos_cos.shape[0],
                                1,
                                rotary_pos_cos.shape[1],
                            ]
                        )
                        rotary_pos_sin = rotary_pos_sin.reshape(
                            [
                                1,
                                rotary_pos_sin.shape[0],
                                1,
                                rotary_pos_sin.shape[1],
                            ]
                        )

            outputs = recompute(
                self._forward_impl,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices.clone()  # Clone is necessary!
                if attn_mask_startend_row_indices is not None
                else None,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb.clone()
                if rotary_pos_emb is not None
                else None,  # Clone is necessary!
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            grid_thws = dict_args.pop("grid_thws", None)
            outputs = self._forward_impl(**dict_args)
            dict_args["grid_thws"] = grid_thws

        if len(outputs) == 3:
            output, context = outputs[0], outputs[1]
        else:
            output, context = outputs, None

        deepstack_feature = outputs[-1]

        rst = OrderedDict()
        rst = {"hidden_states": output}
        if context is not None:
            rst["context"] = context

        rst = {**dict_args, **rst}

        return rst

    def _forward_impl(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor = None,
        attn_mask_startend_row_indices: paddle.Tensor = None,
        context: paddle.Tensor = None,
        context_mask: paddle.Tensor = None,
        rotary_pos_emb: paddle.Tensor = None,
        rotary_pos_cos: paddle.Tensor = None,
        rotary_pos_sin: paddle.Tensor = None,
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
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            in_recompute=self.full_recompute,
        )
        hidden_states = self._forward_mlp(hidden_states)

        if context is not None:
            return hidden_states, context
        return hidden_states
