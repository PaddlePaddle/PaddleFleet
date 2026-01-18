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

import paddle
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
)

from paddlefleet.pipeline_parallel import ScheduleNode
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
        mrope_section: list[int] | None = None,
    ):
        super().__init__(config)
        self.embedding = build_layer(
            sublayers_spec.language_embedding,
            config=config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
        )
        self.sequence_parallel = self.config.sequence_parallel
        if (
            config.num_nextn_predict_layers is not None
            and self.config.num_nextn_predict_layers > 0
            and self.sequence_parallel
        ):
            self.embedding.embed_tokens.reduce_scatter_embeddings = False
            self.embedding.scatter_to_sequence_parallel = False
            self.embedding.reduce_scatter_embeddings = False
            self.embedding.sequence_parallel = False
        self.rotary_pos_emb = None
        self.multimodal_embedding = config.multimodal_embedding
        self.mrope_section = mrope_section
        self.position_embedding_type = position_embedding_type
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

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="GPTEmbedding")

    def forward(
        self,
        dict_args: dict,
        decoder_input: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        input_ids = dict_args["input_ids"]
        position_ids = dict_args.get("position_ids", None)
        position_ids = (
            position_ids.to("gpu") if position_ids is not None else None
        )
        attention_mask = dict_args.get("attention_mask", None)
        attn_mask_startend_row_indices = dict_args.get(
            "attn_mask_startend_row_indices", None
        )
        deepstack_image_embeds = dict_args.get("deepstack_image_embeds", None)
        deepstack_video_embeds = dict_args.get("deepstack_video_embeds", None)
        visual_pos_masks = None
        # Deepstack
        deepstack_visual_embeds = None
        visual_pos_mask = None
        mtp_emb_res = None
        if decoder_input is None:
            decoder_input = self.embedding(
                input_ids=input_ids,
                position_ids=None
                if self.multimodal_embedding
                else position_ids,
            )
            if (
                self.config.num_nextn_predict_layers is not None
                and self.config.num_nextn_predict_layers > 0
            ):
                assert not self.multimodal_embedding, (
                    "MTP not support mm for now."
                )
                inputs_embeds_extra = decoder_input[
                    :, -self.config.num_nextn_predict_layers :, :
                ]  # [B, S, H]
                inputs_embeds = decoder_input[
                    :, : -self.config.num_nextn_predict_layers, :
                ]
                inputs_embeds_ori = inputs_embeds
                batch_size, seq_length, hidden_size = inputs_embeds.shape

                if self.sequence_parallel:
                    inputs_embeds = inputs_embeds.reshape(
                        [-1, inputs_embeds.shape[-1]]
                    )
                    inputs_embeds = ScatterOp.apply(inputs_embeds)
                    inputs_embeds = (
                        inputs_embeds.reshape([batch_size, -1, hidden_size])
                        .permute(1, 0, 2)
                        .contiguous()
                    )  # change to [S, B, H]
                mtp_emb_res = [inputs_embeds]
                for depth in range(self.config.num_nextn_predict_layers):
                    inputs_embeds_mtp = paddle.concat(
                        [
                            inputs_embeds_ori[:, (depth + 1) :, :],
                            inputs_embeds_extra[:, : (depth + 1), :],
                        ],
                        axis=1,
                    )
                    if self.sequence_parallel:
                        inputs_embeds_mtp = inputs_embeds_mtp.reshape(
                            [-1, inputs_embeds_mtp.shape[-1]]
                        )
                        inputs_embeds_mtp = ScatterOp.apply(inputs_embeds_mtp)
                        inputs_embeds_mtp = (
                            inputs_embeds_mtp.reshape(
                                [batch_size, -1, hidden_size]
                            )
                            .permute(1, 0, 2)
                            .contiguous()
                        )  # change to [S, B, H]
                    mtp_emb_res.append(inputs_embeds_mtp)

            if self.multimodal_embedding:
                image_embeds = dict_args.get("image_embeds", None)
                video_embeds = dict_args.get("video_embeds", None)
                if image_embeds is not None:
                    image_mask, _ = self.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=decoder_input,
                        image_features=image_embeds,
                    )
                    decoder_input = decoder_input.masked_scatter(
                        image_mask, image_embeds.astype(decoder_input.dtype)
                    )
                    visual_pos_masks = image_mask[..., 0]
                    deepstack_visual_embeds = deepstack_image_embeds

                if video_embeds is not None:
                    _, video_mask = self.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=decoder_input,
                        video_features=video_embeds,
                    )
                    decoder_input = decoder_input.masked_scatter(
                        video_mask, video_embeds.astype(decoder_input.dtype)
                    )
                    visual_pos_masks = video_mask[..., 0]
                    deepstack_visual_embeds = deepstack_video_embeds

                if image_embeds is not None and video_embeds is not None:
                    image_mask = image_mask[..., 0]
                    video_mask = video_mask[..., 0]
                    visual_pos_masks = image_mask | video_mask
                    deepstack_visual_embeds = []
                    image_mask_joint = image_mask[visual_pos_masks]
                    video_mask_joint = video_mask[visual_pos_masks]
                    for img_embed, vid_embed in zip(
                        deepstack_image_embeds, deepstack_video_embeds
                    ):
                        embed_joint = img_embed.new_zeros(
                            visual_pos_masks.sum(), img_embed.shape[-1]
                        ).to(img_embed.device)
                        embed_joint[image_mask_joint, :] = img_embed
                        embed_joint[video_mask_joint, :] = vid_embed
                        deepstack_visual_embeds.append(embed_joint)
        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None

        if (
            self.position_embedding_type == "rope"
            and self.rotary_pos_emb is not None
        ):
            rope_base = decoder_input if mtp_emb_res is None else mtp_emb_res[0]
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                rope_base, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
                position_ids=position_ids,
            )
        elif (
            self.position_embedding_type == "mrope"
            and self.rotary_pos_emb is not None
        ):
            rotary_pos_emb = self.rotary_pos_emb(
                position_ids, self.mrope_section
            )

        if rotary_pos_emb is not None:
            if self.config.apply_rope_fusion:
                rotary_pos_cos = paddle.cos(rotary_pos_emb)
                rotary_pos_sin = paddle.sin(rotary_pos_emb)
            if self.config.sequence_parallel:
                rotary_pos_emb = rotary_pos_emb.transpose(
                    [1, 0, 2, 3]
                ).contiguous()

        preproc_output = {
            "hidden_states": decoder_input,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
            "rotary_pos_emb": rotary_pos_emb,
            "rotary_pos_cos": rotary_pos_cos,
            "rotary_pos_sin": rotary_pos_sin,
            "position_ids": position_ids,
            "deepstack_visual_emb": deepstack_visual_embeds,
            "visual_pos_masks": visual_pos_masks,
        }
        if mtp_emb_res is not None:
            new_output = {"hidden_states": mtp_emb_res[0]}
            assert (
                self.config.num_nextn_predict_layers is not None
                and self.config.num_nextn_predict_layers > 0
            )
            assert len(mtp_emb_res) == self.config.num_nextn_predict_layers + 1
            for i in range(self.config.num_nextn_predict_layers):
                new_output[f"decoder_input_{i}"] = mtp_emb_res[i + 1]
            preproc_output = {**preproc_output, **new_output}

        for key in list(preproc_output.keys()):
            if preproc_output[key] is None:
                preproc_output.pop(key)
        return preproc_output

    def get_placeholder_mask(
        self,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        image_features: Tensor | None = None,
        video_features: Tensor | None = None,
    ):
        """
        Obtain the multimodal placeholder mask from the input and verify whether the number of placeholder tokens matches the length of the multimodal features.
        If the lengths do not match, an error is thrown.
        Args:
            input_ids: Tensor of input token IDs```
            inputs_embeds: input embedding tensor
            image_features: Tensor of image features, optional```
            video_features: Video feature tensor, optional
        Returns:
            tuple: (special_image_mask, special_video_mask) - Mask tensors for image and video tokens
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.embedding(
                paddle.to_tensor(self.config.image_token_id, dtype="int64")
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.embedding(
                paddle.to_tensor(self.config.video_token_id, dtype="int64")
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )

        if (
            image_features is not None
            and inputs_embeds[special_image_mask].numel()
            != image_features.numel()
        ):
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(
            inputs_embeds
        )
        if (
            video_features is not None
            and inputs_embeds[special_video_mask].numel()
            != video_features.numel()
        ):
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask
