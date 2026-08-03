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

"""Kimi-K3 (MoonViT3d) vision patch/position embedding.

Ported from the HuggingFace reference ``Kimi-K3/modeling_kimi_k3.py``:
per-patch ``Conv2D`` projection (the class name says ``3d`` but the kernel is
2D; the temporal axis is carried by ``grid_thw`` and a fixed sincos time
embedding), plus an interpolatable learnable 2D absolute position embedding.

``KimiK3VisionEmbedding`` is the pipeline-facing layer: it consumes the input
dict and emits the dict consumed by the encoder layers, including the shared
2D RoPE ``rope_freqs_cis``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from ...spec_utils import LayerSpec, build_layer
from ...transformer.layer import FleetLayer


@dataclass
class KimiK3VisionEmbeddingSpec:
    """LayerSpecs of the sublayers owned by ``KimiK3VisionEmbedding``."""

    rope_embedding: LayerSpec = None


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


def get_1d_sincos_pos_embed(embed_dim: int, t_size: int):
    grid_t = np.arange(t_size, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)  # (t_size, D)


def _vision_cu_seqlens(grid_thws: paddle.Tensor) -> paddle.Tensor:
    """Exclusive-scan patch boundaries of each media, as ``[N + 1]`` int32."""
    lengths = grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]
    return (
        paddle.concat([paddle.zeros([1], dtype=lengths.dtype), lengths])
        .cumsum(axis=0)
        .astype("int32")
    )


def build_vision_startend_row_indices(grid_thws: paddle.Tensor):
    """Flashmask block-diagonal bounds, ``[1, 1, total_patches, 2]`` int32.

    The HF reference (``MoonViT3dEncoder.forward``) derives ``cu_seqlens`` from
    ``grid_thws`` inside the encoder and attends non-causally within a single
    media, so the bounds must not be left to the caller.
    """
    cu_seqlens = _vision_cu_seqlens(grid_thws)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    # Rows [start_i, end_i) mask everything from end_i downwards and everything
    # above start_i, i.e. a block-diagonal (packed) mask.
    lower_start = paddle.repeat_interleave(cu_seqlens[1:], lengths).reshape(
        [1, 1, -1, 1]
    )
    upper_end = paddle.repeat_interleave(cu_seqlens[:-1], lengths).reshape(
        [1, 1, -1, 1]
    )
    return paddle.concat([lower_start, upper_end], axis=-1)


def build_vision_block_diag_mask(grid_thws: paddle.Tensor):
    """Dense per-media mask, ``[1, 1, L, L]`` bool where True means masked out.

    The dense ``DotProductAttention`` branch ignores
    ``attn_mask_startend_row_indices``, so float32 / eager needs this instead.
    """
    cu_seqlens = _vision_cu_seqlens(grid_thws)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    media_ids = paddle.repeat_interleave(
        paddle.arange(lengths.shape[0]), lengths
    )
    same_media = media_ids.unsqueeze(0) == media_ids.unsqueeze(1)
    return paddle.logical_not(same_media).unsqueeze(0).unsqueeze(0)


def merge_vision_block_diag_mask(attention_mask, grid_thws: paddle.Tensor):
    """OR the per-media isolation into a caller-supplied dense mask."""
    block_diag = build_vision_block_diag_mask(grid_thws)
    if attention_mask is None:
        return block_diag
    if attention_mask.dtype != paddle.bool:
        # Collate emits float masks where 1.0 means attend.
        attention_mask = attention_mask < 0.5
    if attention_mask.ndim == 2:
        # [batch, kv] padding mask -> broadcast over the query axis.
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
    elif attention_mask.ndim != 4:
        raise ValueError(
            "Kimi-K3 vision attention_mask must be [batch, kv] or "
            f"[batch, heads, q, kv], got {list(attention_mask.shape)}"
        )
    return paddle.logical_or(attention_mask, block_diag)


class Learnable2DInterpPosEmbDivided(nn.Layer):
    """Learnable 2D position embedding interpolated to arbitrary (h, w), plus a
    fixed sincos temporal embedding applied when ``t > 1``.
    """

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bilinear",
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode

        self.weight = self.create_parameter(
            shape=[height, width, dim],
            default_initializer=nn.initializer.Normal(),
        )
        time_weight = get_1d_sincos_pos_embed(dim, num_frames)  # (T, D)
        self.register_buffer(
            "time_weight",
            paddle.to_tensor(time_weight, dtype="float32").unsqueeze(1),
            persistable=False,
        )  # (T, 1, D)

    def _interp(self, h: int, w: int) -> paddle.Tensor:
        # weight: (H, W, D) -> (1, D, H, W) -> interpolate -> (h*w, D)
        org = self.weight.transpose([2, 0, 1]).unsqueeze(0)
        out = F.interpolate(org, size=[h, w], mode=self.interpolation_mode)
        out = out.squeeze(0).transpose([1, 2, 0])  # (h, w, D)
        return out.flatten(stop_axis=1)  # (h*w, D)

    def forward(
        self, x: paddle.Tensor, grid_thws: paddle.Tensor
    ) -> paddle.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            t, h, w = int(t), int(h), int(w)
            assert t <= self.num_frames, f"t:{t} > num_frames:{self.num_frames}"
            if (h, w) == (self.height, self.width):
                pos_emb_2d = self.weight.flatten(stop_axis=1)
            else:
                pos_emb_2d = self._interp(h, w)

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).tile(
                    [t, 1, 1]
                ) + self.time_weight[0:t].astype(pos_emb_2d.dtype)

            pos_embs.append(pos_emb_3d.reshape([-1, pos_emb_3d.shape[-1]]))

        return x + paddle.concat(pos_embs).astype(x.dtype)


class MoonVision3dPatchEmbed(nn.Layer):
    """Conv2D patch projection followed by the absolute position embedding."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple = (14, 14),
        pos_emb_height: int = 64,
        pos_emb_width: int = 64,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        patch_embed_proj_bias: bool = False,
        pos_emb_interpolation_mode: str = "bilinear",
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = tuple(patch_size)

        self.proj = nn.Conv2D(
            in_dim,
            out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias_attr=patch_embed_proj_bias if patch_embed_proj_bias else False,
        )

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
                interpolation_mode=pos_emb_interpolation_mode,
            )
        else:
            raise NotImplementedError(
                f"Unsupported pos_emb_type: {pos_emb_type}"
            )

    def forward(
        self, x: paddle.Tensor, grid_thws: paddle.Tensor
    ) -> paddle.Tensor:
        """x: (L, in_dim, patch, patch) -> (L, out_dim)."""
        x = self.proj(x).reshape([x.shape[0], -1])
        return self.pos_emb(x, grid_thws)


class KimiK3VisionEmbedding(FleetLayer):
    """Pipeline-facing embedding stage of the Kimi-K3 vision tower."""

    def __init__(self, config, sublayers_spec: KimiK3VisionEmbeddingSpec):
        super().__init__(config)

        patch_size = getattr(config, "patch_size", 14)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2:
            raise ValueError(
                f"patch_size must have two dimensions, got {patch_size}"
            )

        self.patch_size = tuple(patch_size)
        self.in_channels = getattr(config, "in_channels", 3)
        self.embed_dim = config.hidden_size
        self.embedding = MoonVision3dPatchEmbed(
            out_dim=self.embed_dim,
            in_dim=self.in_channels,
            patch_size=self.patch_size,
            pos_emb_height=getattr(config, "init_pos_emb_height", 64),
            pos_emb_width=getattr(config, "init_pos_emb_width", 64),
            pos_emb_time=getattr(config, "init_pos_emb_time", 4),
            pos_emb_type=getattr(config, "pos_emb_type", "divided_fixed"),
            patch_embed_proj_bias=getattr(
                config, "patch_embed_proj_bias", False
            ),
            pos_emb_interpolation_mode=getattr(
                config, "pos_emb_interpolation_mode", "bilinear"
            ),
        )

        assert sublayers_spec.rope_embedding is not None, (
            "KimiK3VisionEmbedding requires a rope_embedding spec"
        )
        self.rotary_pos_emb = build_layer(sublayers_spec.rope_embedding)

        # Conv2D / the position embedding are plain paddle layers, so they are
        # created with the default dtype and need an explicit cast.
        if config.params_dtype is not None:
            self.embedding.to(dtype=config.params_dtype)

    @staticmethod
    def _get_grid_thw(dict_args: dict) -> paddle.Tensor:
        grid_thw = dict_args.get("grid_thws")
        if grid_thw is None:
            grid_thw = dict_args.get("grid_thw")
        if grid_thw is None:
            raise KeyError(
                "KimiK3VisionEmbedding requires grid_thws or grid_thw"
            )
        return grid_thw

    def _check_inputs(self, pixel_values, grid_thw):
        if pixel_values.ndim != 4:
            raise ValueError(
                "Kimi-K3 pixel_values must have shape [L, C, patch_h, patch_w], "
                f"got {list(pixel_values.shape)}"
            )
        if pixel_values.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {pixel_values.shape[1]}"
            )
        if tuple(pixel_values.shape[2:]) != self.patch_size:
            raise ValueError(
                f"Expected pre-cut patches of size {self.patch_size}, "
                f"got {tuple(pixel_values.shape[2:])}"
            )
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError(
                f"grid_thw must have shape [N, 3], got {list(grid_thw.shape)}"
            )

        expected_patches = sum(
            int(t) * int(h) * int(w) for t, h, w in grid_thw.tolist()
        )
        if expected_patches != pixel_values.shape[0]:
            raise ValueError(
                f"grid_thw describes {expected_patches} patches, "
                f"but pixel_values contains {pixel_values.shape[0]}"
            )

    def _uses_dense_attention(self, dtype) -> bool:
        """Whether ``DotProductAttention`` will take its dense (eager) branch."""
        return getattr(
            self.config, "_attn_implementation", None
        ) == "eager" or dtype not in (paddle.bfloat16, paddle.float16)

    def forward(self, dict_args: dict) -> dict:
        pixel_values = dict_args["pixel_values"]
        grid_thws = self._get_grid_thw(dict_args)
        self._check_inputs(pixel_values, grid_thws)

        hidden_states = self.embedding(pixel_values, grid_thws)
        rope_freqs_cis = self.rotary_pos_emb.get_freqs_cis(grid_thws)

        startend_row_indices = dict_args.get("attn_mask_startend_row_indices")
        attention_mask = dict_args.get("attention_mask")
        if self._uses_dense_attention(hidden_states.dtype):
            # The dense branch ignores startend_row_indices, so the isolation
            # has to live in attention_mask even when the caller passed bounds.
            attention_mask = merge_vision_block_diag_mask(
                attention_mask, grid_thws
            )
        elif startend_row_indices is None:
            # A caller-supplied attention_mask alone leaves flashmask without
            # per-media bounds, so fill them in regardless of attention_mask.
            startend_row_indices = build_vision_startend_row_indices(grid_thws)

        return {
            "hidden_states": hidden_states.unsqueeze(0),
            "grid_thws": grid_thws,
            "attention_mask": attention_mask,
            "attn_mask_startend_row_indices": startend_row_indices,
            "rope_freqs_cis": rope_freqs_cis,
        }
