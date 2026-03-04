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
from ...transformer.enums import ModelType
from ...transformer.layer import FleetLayer
from ...transformer.transformer_config import TransformerConfig
from ...transformer.transformer_encoder import TransformerEncoder
from ...transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSublayersSpec,
)


def get_image_sequence_length(
    img_h, img_w, patch_dim, add_class_token, class_token_len
):
    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches + (class_token_len if add_class_token else 0)


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
        deepstack_features_list = dict_args.pop("deepstack_features_list", None)
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

        context, deepstack_feature = None, None
        hidden_states = outputs[0]
        if len(outputs) > 1:
            deepstack_feature = outputs[-1]
            if len(outputs) == 3:
                context = outputs[1]

        rst = OrderedDict()
        rst = {"hidden_states": hidden_states}
        if context is not None:
            rst["context"] = context
        if deepstack_features_list is None:
            deepstack_features_list = []
        if deepstack_feature is not None:
            deepstack_features_list.append(deepstack_feature)
        rst["deepstack_features_list"] = deepstack_features_list
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
            deepstack_feature = self.deepstack_merger(
                {"hidden_states": hidden_states}
            )["hidden_states"]

        res = (hidden_states,)
        if context is not None:
            res += (context,)
        if deepstack_feature is not None:
            res += (deepstack_feature,)
        return res


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
        deepstack_visual_emb = dict_args.get("deepstack_visual_emb", None)
        visual_pos_masks = dict_args.get("visual_pos_masks", None)

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

        if isinstance(outputs, tuple):
            output, context = outputs[0], outputs[1]
        else:
            output, context = outputs, None

        # Apply deepstack visual embedding outside of recompute to avoid issues
        # with recompute not properly handling list-of-tensors (deepstack_visual_emb)
        if deepstack_visual_emb and self.layer_number in range(
            len(deepstack_visual_emb)
        ):
            output = self._deepstack_process(
                hidden_states=output,
                visual_embeds=deepstack_visual_emb[self.layer_number],
                visual_pos_masks=visual_pos_masks,
            )

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
        **kwargs,
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


class Qwen3VLModelDist(FleetLayer):
    """Qwen3VL Model Base Model Class."""

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        drop_vision_class_token: bool = False,
        vp_stage: int | None = None,
        model_version: str | None = None,
        criterion=False,
    ) -> None:
        super().__init__(config=config)

        language_transformer_config = config.text_config
        vision_transformer_config = config.vision_config
        self.model_version = (
            vision_transformer_config.model_version
            if model_version is None
            else model_version
        )
        self._language_max_sequence_length = (
            language_transformer_config.max_sequence_length
        )
        assert self.model_version is not None

        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vp_stage = vp_stage

        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.image_token_index = config.image_token_id
        self.video_token_index = config.video_token_id

        self.sequence_parallel_lm = (
            language_transformer_config.sequence_parallel
        )
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = (
            language_transformer_config.context_parallel_size
        )
        assert not (
            self.sequence_parallel_lm or self.context_parallel_lm > 1
        ), (
            f"qwenvl donnot support sequence parallel {self.sequence_parallel_lm} "
            f"or context parallel {self.context_parallel_lm}"
        )
        self.share_embeddings_and_output_weights = False
        self.rope_deltas = None

        if self.add_decoder:
            self.language_model = language_transformer_config.provide(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            )
            self._language_is_pipeline_parallel = (
                language_transformer_config.pipeline_model_parallel_size > 1
            )

        if self.add_encoder:
            self.vision_model = vision_transformer_config.provide()
            self._drop_vision_class_token = drop_vision_class_token

        self.model_type = ModelType.encoder_or_decoder

        self._img_seq_len = get_image_sequence_length(
            img_h=vision_transformer_config.img_h,
            img_w=vision_transformer_config.img_w,
            patch_dim=vision_transformer_config.patch_size,
            add_class_token=not drop_vision_class_token,
            class_token_len=vision_transformer_config.class_token_len,
        )
        self.criterion = criterion

    def get_rope_index(
        self,
        input_ids: paddle.LongTensor | None = None,
        image_grid_thw: paddle.LongTensor | None = None,
        video_grid_thw: paddle.LongTensor | None = None,
        attention_mask: paddle.Tensor | None = None,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = paddle.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        # TODO when implemented data file.
        image_token_id = self.image_token_index
        video_token_id = self.video_token_index
        vision_start_token_id = 151652
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = paddle.ones_like(total_input_ids)
            position_ids = paddle.ones(
                [3, input_ids.shape[0], input_ids.shape[1]],
                dtype=input_ids.dtype,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = paddle.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if llm_pos_ids_list
                        else 0
                    )
                    llm_pos_ids_list.append(
                        paddle.arange(text_len).view(1, -1).expand(3, -1)
                        + st_idx
                    )

                    t_index = (
                        paddle.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        paddle.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        paddle.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        paddle.stack([t_index, h_index, w_index])
                        + text_len
                        + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        paddle.arange(text_len).view(1, -1).expand(3, -1)
                        + st_idx
                    )

                llm_positions = paddle.cat(llm_pos_ids_list, dim=1).reshape(
                    3, -1
                )
                position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = paddle.to_tensor(
                mrope_position_deltas
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = (
                    max_position_ids + 1 - attention_mask.shape[-1]
                )
            else:
                position_ids = (
                    paddle.arange(input_ids.shape[1])
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = paddle.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def get_video_features(
        self,
        pixel_values_videos: paddle.FloatTensor,
        video_grid_thw: paddle.LongTensor | None = None,
    ):
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: paddle.FloatTensor,
        image_grid_thw: paddle.LongTensor | None = None,
    ):
        dict_args = {
            "pixel_values": pixel_values,
            "grid_thw": image_grid_thw,
        }
        vision_output = self.vision_model(dict_args)
        image_embeds, deepstack_image_embeds = (
            vision_output["hidden_states"],
            vision_output["deepstack_features_list"],
        )
        split_sizes = (
            image_grid_thw.prod(-1)
            // self.config.vision_config.spatial_merge_size**2
        ).tolist()
        image_embeds = paddle.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def forward(
        self,
        input_ids: paddle.LongTensor = None,
        attention_mask: paddle.Tensor | None = None,
        position_ids: paddle.LongTensor | None = None,
        loss_mask: paddle.Tensor | None = None,
        labels: paddle.Tensor | None = None,
        inference_params=None,
        pixel_values: paddle.Tensor | None = None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        runtime_gather_output: bool | None = None,
        cache_position: paddle.Tensor | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        **kwargs,
    ) -> paddle.Tensor:
        assert loss_mask is None, "loss_mask is not supported yet"
        (
            image_embeds,
            video_embeds,
            deepstack_image_embeds,
            deepstack_video_embeds,
        ) = (None for _ in range(4))
        if self.add_encoder and pixel_values is not None:
            pixel_values = pixel_values.to(
                self.vision_model.parameters()[0].dtype
            )
            image_embeds, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = paddle.cat(image_embeds, dim=0)

        if self.add_encoder and pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(
                self.vision_model.parameters()[0].dtype
            )
            video_embeds, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = paddle.cat(video_embeds, axis=0)

        if position_ids is None:
            if (
                self.rope_deltas is None
                or cache_position is None
                or cache_position[0] == 0
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                position_ids = paddle.arange(seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(
                    3, batch_size, -1
                )
                if cache_position is not None:
                    delta = cache_position[0] + self.rope_deltas
                else:
                    delta = paddle.zeros((batch_size, seq_length))
                delta = delta.repeat_interleave(
                    batch_size // delta.shape[0], axis=1
                )
                position_ids = position_ids + delta
        else:
            if position_ids.shape == input_ids.shape:
                position_ids = position_ids.expand(3, position_ids.shape[0], -1)

        input_dict = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "decoder_input": None,
            "image_embeds": image_embeds,
            "video_embeds": video_embeds,
            "labels": labels,
            "deepstack_image_embeds": deepstack_image_embeds,
            "deepstack_video_embeds": deepstack_video_embeds,
            "runtime_gather_output": runtime_gather_output,
        }
        output = self.language_model(input_dict)

        # print("qwenvl criterion ",self.criterion)
        if labels is None:
            return output
        elif self.criterion is not None:
            # print("qwenvl output loss  ",self.criterion(output, labels))
            return self.criterion(output, labels)
        else:
            return output

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, (
            "input_tensor should only be length 1 for llava"
        )

        if self.add_encoder and self.add_decoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.add_encoder:
            self.vision_model.set_input_tensor(input_tensor[0])
        elif self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    # def get_input_embeddings(self):
    #     return self.language_model.get_input_embeddings()


__all__ = [
    "Qwen3VLTextTransformerLayer",
    "Qwen3VLVisionModel",
    "Qwen3VLVisionTransformerLayer",
    "Qwen3VLModelDist",
]
