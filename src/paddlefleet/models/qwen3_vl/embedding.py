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
from dataclasses import dataclass

import paddle
from paddle import nn
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
)
from paddle.nn import functional as F

from ...packed_seq_params import PackedSeqParams
from ...spec_utils import LayerSpec, build_layer
from ...transformer import TransformerConfig
from ...transformer.layer import FleetLayer
from ..gpt.gpt_embedding import GPTEmbedding


@dataclass
class VisionEmbeddingSpec:
    rope_embedding: LayerSpec = None


class VisionEmbedding(FleetLayer):
    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: VisionEmbeddingSpec,
    ):
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = (
            self.spatial_merge_size * self.spatial_merge_size
        )
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        self.merge_hidden_size = self.embed_dim * (config.spatial_merge_size**2)

        kernel_size = [
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        ]
        self.patch_embed = nn.Conv3D(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.rotary_pos_emb = None
        if sublayers_spec.rope_embedding:
            self.rotary_pos_emb = build_layer(
                sublayers_spec.rope_embedding,
            )

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = paddle.arange(h).unsqueeze(1).expand([-1, w])
            hpos_ids = hpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            hpos_ids = hpos_ids.transpose(perm=[0, 2, 1, 3])
            hpos_ids = hpos_ids.flatten()

            wpos_ids = paddle.arange(w).unsqueeze(0).expand([h, -1])
            wpos_ids = wpos_ids.reshape(
                [
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                ]
            )
            wpos_ids = wpos_ids.transpose([0, 2, 1, 3])
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                paddle.stack(x=[hpos_ids, wpos_ids], axis=-1).tile(
                    repeat_times=[t, 1]
                )
            )
        pos_ids = paddle.cat(x=pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_axis=1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = (
            grid_thw[:, 0],
            grid_thw[:, 1],
            grid_thw[:, 2],
        )
        device = paddle.get_device()

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = paddle.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = paddle.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(
                max=self.num_grid_per_side - 1
            )
            w_idxs_ceil = (w_idxs.int() + 1).clip(
                max=self.num_grid_per_side - 1
            )

            dh = h_idxs - h_idxs_floor.astype("float32")
            dw = w_idxs - w_idxs_floor.astype("float32")

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = paddle.tensor(idx_list, dtype=paddle.long, device=device)
        weight_tensor = paddle.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = (
            pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        )

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(
            patch_pos_embeds, grid_ts, grid_hs, grid_ws
        ):
            pos_embed = pos_embed.repeat([t, 1])
            pos_embed = (
                pos_embed.view(
                    [
                        t,
                        h // merge_size,
                        merge_size,
                        w // merge_size,
                        merge_size,
                        -1,
                    ]
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = paddle.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def get_packed_seq_params(
        self,
        grid_thw: paddle.Tensor,
    ):
        seqlens = paddle.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).contiguous()
        cu_seqlens = seqlens.cumsum(dim=0, dtype=paddle.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).contiguous()
        cu_seqlens = cu_seqlens.squeeze().contiguous()

        max_seqlen = seqlens.max().item()

        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

    def forward(self, dict_args: dict):
        pixel_values = dict_args["pixel_values"]
        grid_thw = dict_args["grid_thw"]

        hidden_states = self.patch_embed(pixel_values).view()
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape([seq_len, -1])
        hidden_states = hidden_states.unsqueeze(0)

        rotary_pos_emb = self.rotary_pos_emb(grid_thw)
        rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb.get_cos_sin(
            grid_thw
        )

        packed_seq_params = self.get_packed_seq_params(grid_thw)

        preproc_output = {
            "hidden_states": hidden_states,
            "attention_mask": dict_args.get("attention_mask", None),
            "rotary_pos_emb": rotary_pos_emb,
            "rotary_pos_cos": rotary_pos_cos,
            "rotary_pos_sin": rotary_pos_sin,
            "packed_seq_params": packed_seq_params,
        }

        return preproc_output


class TextEmbedding(GPTEmbedding):
    def forward(
        self,
        dict_args: dict,
        decoder_input: paddle.Tensor = None,
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
        # Deepstack
        deepstack_visual_embeds = None
        visual_pos_masks = None
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
            assert (
                self.config.num_nextn_predict_layers is not None
                and self.config.num_nextn_predict_layers > 0
            )
            assert len(mtp_emb_res) == self.config.num_nextn_predict_layers + 1
            hidden_states_concat = paddle.concat(mtp_emb_res)
            preproc_output["hidden_states"] = hidden_states_concat

        for key in list(preproc_output.keys()):
            if preproc_output[key] is None:
                preproc_output.pop(key)
        return preproc_output
