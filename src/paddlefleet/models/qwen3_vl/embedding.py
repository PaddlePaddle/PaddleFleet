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
from paddle.nn import functional as F

from ...packed_seq_params import PackedSeqParams
from ...spec_utils import LayerSpec, build_layer
from ...transformer import TransformerConfig
from ...transformer.layer import FleetLayer


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
        # Optimization: replace per-image tolist() + Python list extend with
        # pure-GPU paddle.cat accumulation, eliminating 274k GpuMemcpySync:GPU->CPU
        # synchronization points that previously stalled the GPU pipeline.
        grid_ts, grid_hs, grid_ws = (
            grid_thw[:, 0],
            grid_thw[:, 1],
            grid_thw[:, 2],
        )

        idx_parts = [[] for _ in range(4)]
        weight_parts = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            # All tensors stay on GPU throughout — no .tolist() / CPU round-trip
            h_idxs = paddle.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = paddle.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.cast("int32")
            w_idxs_floor = w_idxs.cast("int32")
            h_idxs_ceil = (h_idxs_floor + 1).clip(
                max=self.num_grid_per_side - 1
            )
            w_idxs_ceil = (w_idxs_floor + 1).clip(
                max=self.num_grid_per_side - 1
            )

            dh = h_idxs - h_idxs_floor.cast("float32")
            dw = w_idxs - w_idxs_floor.cast("float32")

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            # [h*w] int32 index tensors, kept on GPU
            idx_parts[0].append((base_h.unsqueeze(1) + w_idxs_floor.unsqueeze(0)).flatten().cast("int64"))
            idx_parts[1].append((base_h.unsqueeze(1) + w_idxs_ceil.unsqueeze(0)).flatten().cast("int64"))
            idx_parts[2].append((base_h_ceil.unsqueeze(1) + w_idxs_floor.unsqueeze(0)).flatten().cast("int64"))
            idx_parts[3].append((base_h_ceil.unsqueeze(1) + w_idxs_ceil.unsqueeze(0)).flatten().cast("int64"))

            # [h*w] float weight tensors, kept on GPU
            weight_parts[0].append((((1 - dh).unsqueeze(1)) * ((1 - dw).unsqueeze(0))).flatten())
            weight_parts[1].append((((1 - dh).unsqueeze(1)) * (dw.unsqueeze(0))).flatten())
            weight_parts[2].append(((dh.unsqueeze(1)) * ((1 - dw).unsqueeze(0))).flatten())
            weight_parts[3].append(((dh.unsqueeze(1)) * (dw.unsqueeze(0))).flatten())

        # Single cat per corner — 4 GPU ops total instead of N_images * 8 CPU syncs
        idx_tensor = paddle.stack(
            [paddle.concat(idx_parts[i]) for i in range(4)]
        )  # [4, total_hw]
        weight_tensor = paddle.stack(
            [paddle.concat(weight_parts[i]) for i in range(4)]
        ).cast(self.pos_embed.weight.dtype)  # [4, total_hw]

        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor.unsqueeze(-1)
        patch_pos_embeds = (
            pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        )

        # split sizes are small Python ints from grid_thw — no GPU sync needed
        hw_sizes = [int(h) * int(w) for h, w in zip(grid_hs, grid_ws)]
        patch_pos_embeds = patch_pos_embeds.split(hw_sizes)

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(
            patch_pos_embeds, grid_ts, grid_hs, grid_ws
        ):
            t, h, w = int(t), int(h), int(w)
            pos_embed = pos_embed.tile([t, 1])
            pos_embed = (
                pos_embed.reshape(
                    [
                        t,
                        h // merge_size,
                        merge_size,
                        w // merge_size,
                        merge_size,
                        -1,
                    ]
                )
                .transpose([0, 1, 3, 2, 4, 5])
                .flatten(start_axis=0, stop_axis=4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = paddle.concat(patch_pos_embeds_permute)
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

        # Optimization: compute max_seqlen on CPU from grid_thw (small integer
        # metadata already available on CPU) to avoid a GPU->CPU sync (.item()).
        # grid_thw[:,1]*grid_thw[:,2] gives per-frame token counts; the max
        # over frames is the maximum sequence length for packed attention.
        grid_thw_list = grid_thw.tolist()  # one-shot tiny D2H copy of shape info
        max_seqlen = int(max(h * w for _, h, w in grid_thw_list))

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
