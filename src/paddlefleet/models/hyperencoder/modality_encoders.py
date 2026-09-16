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

"""Image / audio modality towers for HyperEncoder.

This module defines the small vision and audio front-ends that turn raw
patches / mel-spectrogram frames into a shared embedding space, plus the
projector that maps those embeddings into the language model width.

Design notes:

* ``MlpProjector`` exposes its submodule as the attribute ``layers`` because
  the checkpoint weight keys are ``projector.layers.weight``. This name is kept
  consistent across all projector types (including the ``nn.Sequential`` case).
* ``ImageEncoderConv`` accepts an ``out_chans`` argument that it does not use.
  It is retained only to keep the constructor signature stable.
* ``AudioEncoderConv`` returns a 4-D channel-last tensor ``[B, 1, T', 768]``
  (``h'=1, w'=T'``) so it is structurally identical to the image tower output
  ``[B, h', w', 768]``. Downstream code relies on this shape equivalence to
  reuse a single ``permute(0,3,1,2).flatten(2).transpose(1,2)`` path for both
  modalities.

Numerical details:

* ``F.interpolate(align_corners=False)`` is used for position-embedding
  resampling. Both interpolation helpers short-circuit when the source and
  target sizes already match, returning the input unchanged.
* ``F.gelu`` is called with ``approximate=False`` explicitly so it uses the
  exact erf formulation rather than the tanh approximation.
"""

from __future__ import annotations

import paddle
import paddle.nn.functional as F
from paddle import nn

__all__ = [
    "MlpProjector",
    "PatchEmbed",
    "ImageEncoderConv",
    "AudioEncoderConv",
    "get_abs_pos_2d",
    "get_abs_pos_1d",
]


class MlpProjector(nn.Layer):
    """Modality-level projector: 768 -> 1280. Shared by the image and audio towers.

    Only ``projector_type="linear"`` is used in practice; the ``identity`` and
    ``mlp_gelu`` variants are kept for completeness. All variants store their
    submodule under ``self.layers`` so the checkpoint weight keys stay stable.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        projector_type = cfg["projector_type"]
        input_dim = cfg["input_dim"]
        n_embed = cfg["n_embed"]
        depth = cfg.get("depth", 1)

        if projector_type == "identity":
            modules: nn.Layer = nn.Identity()
        elif projector_type == "linear":
            modules = nn.Linear(input_dim, n_embed)
        elif projector_type == "mlp_gelu":
            layers = [nn.Linear(input_dim, n_embed)]
            for _ in range(1, depth):
                layers.append(nn.GELU())
                layers.append(nn.Linear(n_embed, n_embed))
            modules = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        # Attribute name must be ``layers``; weight key is ``projector.layers.weight``.
        self.layers = modules

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply ``self.layers(x)``."""
        return self.layers(x)


class PatchEmbed(nn.Layer):
    """Image -> patch embedding via a single Conv2d."""

    def __init__(
        self,
        kernel_size: tuple[int, int] = (14, 14),
        stride: tuple[int, int] = (14, 14),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        return x.transpose([0, 2, 3, 1])


def get_abs_pos_2d(
    abs_pos: paddle.Tensor, tgt_size: tuple[int, int]
) -> paddle.Tensor:
    """Resample a 2-D positional embedding to the actual patch grid.

    When the source grid size already equals ``tgt_size`` the input is returned
    unchanged (short-circuit). Otherwise the embedding is transposed to
    channel-first, cast to fp32, resized with bilinear interpolation
    (``align_corners=False``), cast back and transposed to channel-last.
    """
    src_size = (int(abs_pos.shape[1]), int(abs_pos.shape[2]))
    if src_size != tuple(int(v) for v in tgt_size):
        old = abs_pos.transpose([0, 3, 1, 2]).astype("float32")
        new = F.interpolate(
            old,
            size=[int(tgt_size[0]), int(tgt_size[1])],
            mode="bilinear",
            align_corners=False,
        ).astype(abs_pos.dtype)
        return new.transpose([0, 2, 3, 1])
    return abs_pos


def get_abs_pos_1d(abs_pos: paddle.Tensor, tgt_size: int) -> paddle.Tensor:
    """Resample a 1-D positional embedding. Short-circuits when sizes match."""
    src_size = int(abs_pos.shape[1])
    if src_size != int(tgt_size):
        old = abs_pos.transpose([0, 2, 1]).astype("float32")
        new = F.interpolate(
            old, size=[int(tgt_size)], mode="linear", align_corners=False
        )
        return new.astype(abs_pos.dtype).transpose([0, 2, 1])
    return abs_pos


class ImageEncoderConv(nn.Layer):
    """Patch embedding plus a learnable 2-D positional embedding.

    Default constructor arguments:
    ``img_size=728, patch_size=14, in_chans=3, embed_dim=768, out_chans=256``.

    ``out_chans`` is accepted but never used; it is kept to preserve a stable
    constructor signature.
    """

    def __init__(
        self,
        img_size: int = 728,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        out_chans: int = 256,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        grid = img_size // patch_size  # 728 // 14 = 52
        # Learnable parameter (not a buffer), initialized to all zeros.
        self.pos_embed = self.create_parameter(
            shape=[1, grid, grid, embed_dim],
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """``[B, 3, H, W]`` -> ``[B, H//14, W//14, 768]``."""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos_2d(self.pos_embed, (x.shape[1], x.shape[2]))
        return x


class AudioEncoderConv(nn.Layer):
    """Mel-spectrogram convolutions plus a learnable 1-D positional embedding.

    Default constructor arguments: ``num_mel_bins=128, embed_dim=768``.

    ``conv2`` uses ``stride=2``, which is why the output length is
    ``T' = (T+1)//2``. ``max_position_embeddings=1500`` sets the positional
    embedding length.
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        embed_dim: int = 768,
        max_position_embeddings: int = 1500,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1D(
            num_mel_bins, embed_dim, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1D(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.pos_embed = self.create_parameter(
            shape=[1, max_position_embeddings, embed_dim],
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, input_features: paddle.Tensor) -> paddle.Tensor:
        """``[B, 128, T]`` -> ``[B, 1, T', 768]``, with ``T' = (T+1)//2``.

        The output is deliberately 4-D channel-last (``h'=1, w'=T'``) so it
        matches the image tower's output shape.

        Note: the conv layers must accumulate in fp32 for numerical stability.
        """
        # approximate=False forces the exact erf GELU rather than the tanh form.
        x = F.gelu(self.conv1(input_features), approximate=False)
        x = F.gelu(self.conv2(x), approximate=False)
        x = x.transpose([0, 2, 1])  # [B, T', embed_dim]
        x = x + get_abs_pos_1d(self.pos_embed, x.shape[1])
        return x.unsqueeze(1)  # [B, 1, T', embed_dim]
