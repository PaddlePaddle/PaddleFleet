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
from ...tensor_parallel.mappings import (
    scatter_to_sequence_parallel_region,
)
from ...transformer import TransformerConfig
from ...transformer.layer import FleetLayer
from ..gpt.gpt_embedding import GPTEmbedding


def safe_repeat_interleave_values(values, repeats):
    max_repeats = paddle.max(repeats)
    mask = paddle.arange(max_repeats).unsqueeze(0) < repeats.unsqueeze(1)
    expanded_values = values.unsqueeze(1).expand([values.shape[0], max_repeats])
    result = paddle.masked_select(expanded_values, mask)
    return result


@dataclass
class VisionEmbeddingSpec:
    rope_embedding: LayerSpec = None


class VisionRotaryEmbedding(nn.Layer):
    inv_freq: paddle.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (paddle.arange(0, dim, 2, dtype=paddle.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def forward(self, seqlen: int) -> paddle.Tensor:
        seq = paddle.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = paddle.outer(seq, self.inv_freq)
        return freqs


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

    def _build_token_image_mapping(self, grid_thw):
        """Build token-to-image mapping, shared by rot_pos_emb and fast_pos_embed_interpolate"""
        heights = grid_thw[:, 1]
        widths = grid_thw[:, 2]
        frames = grid_thw[:, 0]

        num_tokens = frames * heights * widths  # [N]

        total_tokens = num_tokens.sum().item()  # 1 D2H
        max_hw = paddle.max(paddle.maximum(heights, widths)).item()  # 1 D2H

        # token-to-image mapping: image_id[j] = i, where cu_tokens[i] <= j < cu_tokens[i+1]
        cu_tokens = paddle.concat(
            [paddle.zeros([1], dtype="int64"), num_tokens.cumsum(0)]
        )
        global_idx = paddle.arange(total_tokens, dtype="int64")
        image_id = (
            global_idx.unsqueeze(-1) >= cu_tokens[:-1].unsqueeze(0)
        ).astype("int64").sum(-1) - 1

        local_idx = global_idx - cu_tokens[image_id]

        # frame-local index
        token_hw = (heights * widths)[image_id]
        frame_local_idx = local_idx % token_hw

        return image_id, frame_local_idx, total_tokens, max_hw

    def rot_pos_emb(
        self,
        grid_thw,
        image_id=None,
        frame_local_idx=None,
        total_tokens=None,
        max_hw=None,
    ):
        m = self.spatial_merge_size
        widths = grid_thw[:, 2]
        merged_w = widths // m

        if image_id is None:
            image_id, frame_local_idx, total_tokens, max_hw = (
                self._build_token_image_mapping(grid_thw)
            )

        freq_table = self.rotary_pos_emb(max_hw)

        token_mw = merged_w[image_id]  # [total_tokens]

        # Decompose linear index to coordinates: layout [merged_h, merged_w, m, m]
        mm = m * m
        mw_mm = token_mw * mm
        block_row = frame_local_idx // mw_mm
        r1 = frame_local_idx % mw_mm
        block_col = r1 // mm
        r2 = r1 % mm
        intra_row = r2 // m
        intra_col = r2 % m

        row_idx = block_row * m + intra_row
        col_idx = block_col * m + intra_col

        pos_ids = paddle.stack([row_idx, col_idx], axis=-1)  # [total_tokens, 2]

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(start_axis=1)
        return embeddings

    def fast_pos_embed_interpolate(
        self,
        grid_thw,
        image_id=None,
        frame_local_idx=None,
        total_tokens=None,
        max_hw=None,
    ):
        N = self.num_grid_per_side
        m = self.spatial_merge_size
        heights = grid_thw[:, 1]
        widths = grid_thw[:, 2]
        merged_w = widths // m

        if image_id is None:
            image_id, frame_local_idx, total_tokens, max_hw = (
                self._build_token_image_mapping(grid_thw)
            )

        token_mw = merged_w[image_id]

        # Decompose linear index to coordinates (same layout as rot_pos_emb)
        mm = m * m
        mw_mm = token_mw * mm
        block_row = frame_local_idx // mw_mm
        r1 = frame_local_idx % mw_mm
        block_col = r1 // mm
        r2 = r1 % mm
        intra_row = r2 // m
        intra_col = r2 % m

        # Pixel coordinates
        j_h = (block_row * m + intra_row).astype("float32")
        j_w = (block_col * m + intra_col).astype("float32")

        # Bilinear interpolation: h_idx = j_h * (N-1) / (h-1)
        token_h = heights[image_id].astype("float32")
        token_w = widths[image_id].astype("float32")
        h_denom = (token_h - 1).clip(min=1.0)
        w_denom = (token_w - 1).clip(min=1.0)
        h_idx = j_h * (N - 1) / h_denom
        w_idx = j_w * (N - 1) / w_denom

        h_floor = h_idx.astype("int32")
        w_floor = w_idx.astype("int32")
        h_ceil = (h_floor + 1).clip(max=N - 1)
        w_ceil = (w_floor + 1).clip(max=N - 1)

        dh = h_idx - h_floor.astype("float32")
        dw = w_idx - w_floor.astype("float32")

        base_h = h_floor * N
        base_h_ceil = h_ceil * N

        idx_tensor = paddle.stack(
            [
                (base_h + w_floor).astype("int64"),
                (base_h + w_ceil).astype("int64"),
                (base_h_ceil + w_floor).astype("int64"),
                (base_h_ceil + w_ceil).astype("int64"),
            ]
        )  # [4, total_tokens]

        weight_tensor = paddle.stack(
            [(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw]
        ).astype(self.pos_embed.weight.dtype)  # [4, total_tokens]

        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = (
            pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        )
        # Already in (block_h, block_w, intra_h, intra_w) order, no merge_reshape needed
        return patch_pos_embeds

    def get_packed_seq_params(
        self,
        grid_thw: paddle.Tensor,
    ):
        seqlens = safe_repeat_interleave_values(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = seqlens.cumsum(axis=0, dtype=paddle.int32)
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

    def forward(self, dict_args: dict) -> paddle.Tensor:
        pixel_values = dict_args["pixel_values"]
        grid_thw = dict_args["grid_thw"]

        # Pathed embedding
        hidden_states = self.patch_embed(pixel_values).view(-1, self.embed_dim)

        # Share token-to-image mapping to avoid redundant computation
        image_id, frame_local_idx, total_tokens, max_hw = (
            self._build_token_image_mapping(grid_thw)
        )

        pos_embeds = self.fast_pos_embed_interpolate(
            grid_thw,
            image_id=image_id,
            frame_local_idx=frame_local_idx,
            total_tokens=total_tokens,
            max_hw=max_hw,
        )
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape([seq_len, -1])
        hidden_states = hidden_states.unsqueeze(0)

        rotary_pos_emb = self.rot_pos_emb(
            grid_thw,
            image_id=image_id,
            frame_local_idx=frame_local_idx,
            total_tokens=total_tokens,
            max_hw=max_hw,
        )
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        rotary_pos_emb = paddle.cat((rotary_pos_emb, rotary_pos_emb), axis=-1)
        # Cast freqs to float32 and compute cos/sin inside auto_cast(False) to match the
        # precision of _apply_rotary_pos_emb_bshd_fp32, which computes cos/sin on the same
        # bf16 freqs but under auto_cast(False) using a float32 kernel.
        with paddle.amp.auto_cast(False):
            _freqs_f32 = rotary_pos_emb.astype("float32")
            rotary_pos_cos = paddle.cos(_freqs_f32)
            rotary_pos_sin = paddle.sin(_freqs_f32)
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]
        rotary_pos_emb = rotary_pos_emb.transpose([1, 0, 2, 3])

        packed_seq_params = self.get_packed_seq_params(grid_thw)

        # Pre-compute attn_mask_startend_row_indices once for all ViT layers
        cu_seqlens = packed_seq_params.cu_seqlens_kv
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        indices_per_segment = paddle.stack(
            [
                cu_seqlens[1:],  # col 0: lower_start = end_i
                paddle.full_like(
                    cu_seqlens[1:], seq_len
                ),  # col 1: lower_end   = total_seq
                paddle.zeros_like(cu_seqlens[:-1]),  # col 2: upper_start = 0
                cu_seqlens[:-1],  # col 3: upper_end   = start_i
            ],
            axis=1,
        )  # [num_segments, 4]
        attn_mask_startend_row_indices = (
            paddle.repeat_interleave(indices_per_segment, lengths, axis=0)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1, 1, seq_len, 4]

        preproc_output = {
            "hidden_states": hidden_states,
            "attention_mask": dict_args.get("attention_mask", None),
            "rotary_pos_emb": rotary_pos_emb,
            "rotary_pos_cos": rotary_pos_cos,
            "rotary_pos_sin": rotary_pos_sin,
            "packed_seq_params": packed_seq_params,
            "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
        }

        return preproc_output


class Qwen3VLTextEmbedding(GPTEmbedding):
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
                    # Replace masked_scatter with arithmetic blend to avoid
                    # IndexingBackwardKernel (sparse scatter) in the backward pass.
                    #   image_mask : [B, S, H] bool
                    #   image_embeds: [N_img, H]  (N_img = number of image tokens)
                    # Expand image_embeds into the full [B, S, H] space by:
                    #   1. flatten decoder_input and image_mask to 1-D
                    #   2. use paddle.scatter (dense backward = gather) to place
                    #      image_embeds values at the True positions
                    #   3. blend with original decoder_input via mask arithmetic
                    #
                    # Optimization: reuse decoder_input's flattened buffer as the
                    # scatter base (scaled by (1-mask)) to avoid a separate
                    # paddle.zeros([n_total]) allocation (~192 MB bf16 tensor).
                    image_mask_f = image_mask.astype(
                        decoder_input.dtype
                    )  # [B,S,H] float
                    flat_indices = paddle.nonzero(
                        image_mask.reshape([-1])
                    ).squeeze(
                        -1
                    )  # [N_img*H] int64 — dense nonzero, no scatter bwd
                    # Scale the base tensor by (1 - mask) in-place before scatter
                    # so that visual positions are zero — no extra zeros allocation.
                    base_flat = (decoder_input * (1.0 - image_mask_f)).reshape(
                        [-1]
                    )
                    image_src_flat = paddle.scatter(
                        base_flat,
                        flat_indices,
                        image_embeds.astype(decoder_input.dtype).reshape([-1]),
                    )  # scatter bwd is a simple gather — no sparse atomics
                    decoder_input = image_src_flat.reshape(decoder_input.shape)
                    visual_pos_masks = image_mask[..., 0]
                    deepstack_visual_embeds = deepstack_image_embeds

                if video_embeds is not None:
                    _, video_mask = self.get_placeholder_mask(
                        input_ids,
                        inputs_embeds=decoder_input,
                        video_features=video_embeds,
                    )
                    video_mask_f = video_mask.astype(decoder_input.dtype)
                    flat_indices = paddle.nonzero(
                        video_mask.reshape([-1])
                    ).squeeze(-1)
                    base_flat = (decoder_input * (1.0 - video_mask_f)).reshape(
                        [-1]
                    )
                    video_src_flat = paddle.scatter(
                        base_flat,
                        flat_indices,
                        video_embeds.astype(decoder_input.dtype).reshape([-1]),
                    )
                    decoder_input = video_src_flat.reshape(decoder_input.shape)
                    visual_pos_masks = video_mask[..., 0]
                    deepstack_visual_embeds = deepstack_video_embeds

                if image_embeds is not None and video_embeds is not None:
                    image_mask = image_mask[..., 0]  # [B, S] bool
                    video_mask = video_mask[..., 0]  # [B, S] bool
                    visual_pos_masks = image_mask | video_mask
                    deepstack_visual_embeds = []
                    for img_embed, vid_embed in zip(
                        deepstack_image_embeds, deepstack_video_embeds
                    ):
                        # Build embed_joint [N_visual, H] without boolean-index
                        # scatter. Use dense mask arithmetic instead.
                        #   img_embed : [N_img, H]
                        #   vid_embed : [N_vid, H]
                        #   visual_pos_masks: [B, S] bool, N_visual True entries
                        # img_mask_in_visual[i] = True  iff visual position i is image
                        # Computed as: image_mask flattened, keep only visual positions,
                        # expressed as a dense [N_visual] float mask — no indexing.
                        h = img_embed.shape[-1]
                        n_visual = int(visual_pos_masks.sum())
                        # visual_pos_flat: [B*S] bool
                        visual_pos_flat = visual_pos_masks.reshape([-1])
                        image_mask_flat = image_mask.reshape([-1])  # [B*S] bool
                        video_mask_flat = video_mask.reshape([-1])  # [B*S] bool
                        # Dense [B*S] float masks, then compress to [N_visual] via
                        # paddle.masked_select (forward: gather, backward: scatter_add
                        # — but scalar backward is efficient, no sparse atomics)
                        img_mask_in_vis_f = paddle.masked_select(
                            image_mask_flat.astype(img_embed.dtype),
                            visual_pos_flat,
                        ).unsqueeze(-1)  # [N_visual, 1]
                        vid_mask_in_vis_f = paddle.masked_select(
                            video_mask_flat.astype(vid_embed.dtype),
                            visual_pos_flat,
                        ).unsqueeze(-1)  # [N_visual, 1]
                        embed_joint = (
                            img_embed.reshape([n_visual, h]) * img_mask_in_vis_f
                            + vid_embed.reshape([n_visual, h])
                            * vid_mask_in_vis_f
                        )
                        deepstack_visual_embeds.append(embed_joint)
                # Scatter decoder_input to SP format [S/tp, B, H] after multimodal
                # token replacement, since LanguageModelEmbedding's internal scatter
                # was disabled to allow image/video embedding insertion first.
                if self.sequence_parallel:
                    decoder_input = decoder_input.transpose(
                        [1, 0, 2]
                    ).contiguous()
                    decoder_input = scatter_to_sequence_parallel_region(
                        decoder_input, group=self.embedding.tp_group
                    )
                    if self.config.clone_scatter_output_in_embedding:
                        decoder_input = decoder_input.clone()
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


__all__ = ["Qwen3VLTextEmbedding"]
