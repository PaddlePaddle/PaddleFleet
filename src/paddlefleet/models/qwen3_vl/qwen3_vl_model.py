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
from ...spec_utils import LayerSpec, build_layer
from ...transformer.transformer_config import TransformerConfig
from ...transformer.transformer_encoder import TransformerEncoder
from ...transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)


@dataclass
class Qwen3VLVisionSublayersSpec:
    """
    The dataclass for LayerSpecs of Qwen3-VL vision model sublayers_spec,
    including embedding, n * transformer_layer, patch_merger, deepstack_merger.
    """

    embedding: LayerSpec = None
    head_empty_layers: list[LayerSpec] = None
    transformer_layers: list[LayerSpec] = None
    tail_empty_layers: list[LayerSpec] = None
    merger: LayerSpec = None


@dataclass
class Qwen3VLVsisionTransformerSubLayerSpec(TransformerLayerSublayersSpec):
    deepstack_merger: LayerSpec = None


class Qwen3VLVisionModel(TransformerEncoder):
    def get_layer_desc_list(self, spec: Qwen3VLVisionSublayersSpec):
        layers = []
        if self.modal:
            name_prefix = f"model.{self.modal}"
        else:
            name_prefix = "model"

        self.add_sequential_layer(
            layers, LayerDesc(spec.embedding), name_prefix
        )
        self.get_encoder_layer_desc_list(layers, spec, name_prefix)

        self.add_sequential_layer(
            layers, LayerDesc(spec.merger), f"{name_prefix}.merger"
        )

        return layers


class Qwen3VLVisionTransformerLayer(TransformerLayer):
    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: Qwen3VLVsisionTransformerSubLayerSpec,
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
        self.deepstack_merger = None
        if sublayers_spec.deepstack_merger is not None:
            self.deepstack_merger = build_layer(
                sublayers_spec.deepstack_merger,
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
            outputs = self._forward_impl(**dict_args)

        if len(outputs) == 3:
            output, context = outputs[0], outputs[1]
        else:
            output, context = outputs, None

        deepstack_feature = outputs[-1]

        rst = OrderedDict()
        rst = {"hidden_states": output}
        if context is not None:
            rst["context"] = context
        if "deepstack_feature_lists" not in rst:
            rst["deepstack_feature_lists"] = []
        if deepstack_feature is not None:
            rst["deepstack_feature_lists"].append(deepstack_feature)
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
            packed_seq_params=packed_seq_params,
        )
        hidden_states = self._forward_mlp(hidden_states)

        deepstack_feature = None
        if self.deepstack_merger is not None:
            deepstack_feature = self.deepstack_merger(hidden_states)

        if context is not None:
            return hidden_states, context, deepstack_feature
        return hidden_states, deepstack_feature


class Qwen3VLTextTransformerLayer(TransformerLayer):
    """Qwen3VL text model for adapt deepstack process"""

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
            deepstack_visual_emb = dict_args.get("deepstack_visual_emb", None)
            visual_pos_masks = dict_args.get("visual_pos_masks", None)

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
                deepstack_visual_emb=deepstack_visual_emb,
                visual_pos_masks=visual_pos_masks,
            )
        else:
            outputs = self._forward_impl(**dict_args)

        if isinstance(outputs, tuple):
            output, context = outputs[0], outputs[1]
        else:
            output, context = outputs, None

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
        deepstack_visual_emb: list[paddle.Tensor] | None = None,
        visual_pos_masks: paddle.Tensor = None,
    ):
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
            packed_seq_params=packed_seq_params,
        )
        hidden_states = self._forward_mlp(hidden_states)
        if deepstack_visual_emb and self.layer_number in range(
            len(deepstack_visual_emb)
        ):
            # print("process _deepstack_process ",hidden_states.shape,visual_pos_masks.shape,deepstack_visual_emb[self.layer_number].shape)
            hidden_states = self._deepstack_process(
                hidden_states=hidden_states,
                visual_embeds=deepstack_visual_emb[self.layer_number],
                visual_pos_masks=visual_pos_masks,
            )
        if context is not None:
            return hidden_states, context
        return hidden_states

    def _deepstack_process(
        self,
        hidden_states: paddle.Tensor,
        visual_pos_masks: paddle.Tensor,
        visual_embeds: paddle.Tensor,
    ):
        # Store original shape and flatten hidden_states to 2D [B*S, D]
        original_shape = hidden_states.shape
        if hidden_states.ndim > 2:
            hidden_states = hidden_states.flatten(start_axis=0, stop_axis=1)

        visual_embeds = visual_embeds.to(
            hidden_states.device, hidden_states.dtype
        )

        # complicated logic for sequential parallelism
        if visual_pos_masks.ndim > 1:
            visual_pos_masks = visual_pos_masks.flatten()

        # This block handles Sequence Parallelism (Row Slicing)
        if visual_pos_masks.shape[0] > hidden_states.shape[0]:
            try:
                from paddle.distributed.fleet import (
                    get_hybrid_communicate_group,
                )

                hcg = get_hybrid_communicate_group()
                mp_rank = hcg.get_model_parallel_rank()
                mp_size = hcg.get_model_parallel_world_size()
            except (ImportError, AttributeError):
                mp_size = visual_pos_masks.shape[0] // hidden_states.shape[0]
                mp_rank = paddle.distributed.get_rank() % mp_size
            total_len = visual_pos_masks.shape[0]
            chunk_size = total_len // mp_size
            start_idx = mp_rank * chunk_size
            end_idx = start_idx + chunk_size
            if start_idx > 0:
                pre_mask = visual_pos_masks[:start_idx]
                visual_offset = paddle.sum(
                    paddle.cast(pre_mask, "int32")
                ).item()
            else:
                visual_offset = 0
            local_mask = visual_pos_masks[start_idx:end_idx]
            local_visual_count = paddle.sum(
                paddle.cast(local_mask, "int32")
            ).item()

            visual_embeds = visual_embeds[
                visual_offset : visual_offset + local_visual_count
            ]
            visual_pos_masks = local_mask

        # If TP is enabled, hidden_states has shape [..., Hidden_Dim / TP_Size],
        # but visual_embeds usually has full [Hidden_Dim]. We need to slice visual_embeds column-wise.
        if hidden_states.shape[-1] != visual_embeds.shape[-1]:
            try:
                from paddle.distributed.fleet import (
                    get_hybrid_communicate_group,
                )

                hcg = get_hybrid_communicate_group()
                tp_rank = hcg.get_model_parallel_rank()
                tp_size = hcg.get_model_parallel_world_size()
            except (ImportError, AttributeError):
                # Fallback simple estimation
                tp_size = visual_embeds.shape[-1] // hidden_states.shape[-1]
                tp_rank = paddle.distributed.get_rank() % tp_size

            if tp_size > 1:
                embed_dim = visual_embeds.shape[-1]
                slice_width = embed_dim // tp_size
                start_col = tp_rank * slice_width
                end_col = start_col + slice_width
                visual_embeds = visual_embeds[:, start_col:end_col]

        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = (
            local_this  # 这个操作可能会导致paddle转静态图或推理时出问题，建议使用 scatter
        )

        # [Supplement 3] Restore original shape [B*S, D] -> [B, S, D] if necessary
        if len(original_shape) > 2:
            hidden_states = hidden_states.reshape(original_shape)

        return hidden_states
