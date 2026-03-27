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

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

    from paddlefleet.transformer.transformer_config import TransformerConfig
import paddle
from paddle import Tensor
from paddle.incubate.nn.functional import (
    fused_rotary_position_embedding as fused_rope,
)

from paddlefleet.utils import get_pg_rank, get_pg_size

logger = logging.getLogger(__name__)

__all__ = [
    "apply_rotary_pos_emb",
    "get_pos_emb_on_this_cp_rank",
]


def get_pos_emb_on_this_cp_rank(
    pos_emb: Tensor, seq_dim: int, cp_group: Group
) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
        cp_group (Group): The context parallel group
    """
    if cp_group is None:
        raise ValueError(
            "cp_group must be provided to get positional embedding per CP rank"
        )
    cp_size = get_pg_size(cp_group)
    cp_rank = get_pg_rank(cp_group)
    cp_idx = paddle.to_tensor([cp_rank, (2 * cp_size - cp_rank - 1)])
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim],
        2 * cp_size,
        -1,
        *pos_emb.shape[(seq_dim + 1) :],
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :]
    )
    return pos_emb


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = paddle.chunk(x, 2, axis=-1)
        return paddle.cat((-x2, x1), axis=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_new = paddle.stack((-x2, x1), axis=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def get_unsqueeze_dim(t, freqs):
    # x: [b,seq,head_nums,head_dim] or [b,head_nums,seq,head_dim]
    # freqs: [b,seq,head_dim]
    seq_len = freqs.shape[1]
    return 2 if t.shape[1] == seq_len else 1


def _apply_rotary_pos_emb_bshd_fp32(
    t: Tensor,
    t_pass: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        t_pass (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    with paddle.amp.auto_cast(False):
        orig_t_dtype = t.dtype
        t = t.astype(dtype="float32")
        rotate_t = _rotate_half(t, rotary_interleaved)
        cos_ = (paddle.cos(freqs) * mscale).to(t.dtype)
        sin_ = (paddle.sin(freqs) * mscale).to(t.dtype)

        if len(cos_.shape) < len(t.shape):
            # [b,s,h]->[b,s,1,h]
            unsqueeze_dim = get_unsqueeze_dim(t, cos_)
            cos_.unsqueeze_(unsqueeze_dim)
            sin_.unsqueeze_(unsqueeze_dim)
        if len(rotate_t.shape) < len(t.shape):
            rotate_t.reshape_(t.shape)

        t = (t * cos_) + (rotate_t * sin_)
        skip_t_pass = t_pass.shape[-1] == 0
        if not skip_t_pass:
            t_pass = t_pass.astype(dtype="float32")
            res = paddle.cat((t, t_pass), axis=-1).astype(orig_t_dtype)
        else:
            res = t.astype(orig_t_dtype)

        return res


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    high_precision_rope: bool = False,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # For M-RoPE with sequence parallel, freqs may be [S, B, D] while t is [B, S, H, D].
    # When the first two dims are swapped (same product but different order), transpose
    # freqs to align with t's [batch, seq] layout.  A plain reshape would silently
    # reinterpret the memory without reordering data, giving wrong results for B > 1.
    if freqs.ndim == 3:
        t_d0, t_d1 = t.shape[0], t.shape[1]
        f_d0, f_d1 = freqs.shape[0], freqs.shape[1]
        if (t_d0 != f_d0 or t_d1 != f_d1) and t_d0 * t_d1 == f_d0 * f_d1:
            freqs = freqs.transpose([1, 0, 2]).contiguous()

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if high_precision_rope:
        return _apply_rotary_pos_emb_bshd_fp32(
            t,
            t_pass=t_pass,
            freqs=freqs,
            rotary_interleaved=rotary_interleaved,
            mscale=mscale,
        )
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (paddle.cos(freqs) * mscale).to(t.dtype)
    sin_ = (paddle.sin(freqs) * mscale).to(t.dtype)
    if len(cos_.shape) < len(t.shape):
        # [b,s,h]->[b,s,1,h]
        unsqueeze_dim = get_unsqueeze_dim(t, cos_)
        cos_.unsqueeze_(unsqueeze_dim)
        sin_.unsqueeze_(unsqueeze_dim)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return paddle.cat((t, t_pass), axis=-1)


def _get_thd_freqs_on_this_cp_rank(
    cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor, offset: int = 0
) -> Tensor:
    """Get the correct frequency slice for this context parallel rank with optional sequence offset.

    Args:
        cp_rank: Current context parallel rank
        cp_size: Total context parallel size
        x: Input tensor for current sequence
        freqs: Frequency tensor - either full batch positions or max sequence length
        offset: Starting position offset for this sequence in the original batch (default: 0)

    Returns:
        Tensor: Frequency slice corresponding to this CP rank's portion of the sequence

    Note:
        This function supports two modes based on the offset parameter:
        1. offset > 0: Exact mapping mode - freqs contains all positions across all sequences.
           The offset ensures each sequence gets frequencies from its actual position within
           the overall batch. Critical for non-1D RoPE in VLMs where spatial positions matter.
        2. offset = 0: Traditional mode - freqs contains only max sequence length positions.
           All sequences use frequencies starting from position 0, preserving backward
           compatibility.
    """
    if cp_size > 1:
        cp_seg = x.size(1) // 2
        full_seqlen = cp_size * x.size(1)
        # Apply offset to both forward and backward segments for context parallelism
        # offset=0: traditional behavior, freqs[0:cp_seg] and freqs[...]
        # offset>0: exact mapping, freqs[offset+0:offset+cp_seg] and freqs[offset+...]
        return paddle.cat(
            [
                freqs[
                    :,
                    offset + cp_rank * cp_seg : offset + (cp_rank + 1) * cp_seg,
                ],
                freqs[
                    :,
                    offset + full_seqlen - (cp_rank + 1) * cp_seg : offset
                    + full_seqlen
                    - cp_rank * cp_seg,
                ],
            ]
        )
    else:
        # For single context parallel rank:
        # offset=0: use freqs[0:x.size(0)] (traditional)
        # offset>0: use freqs[offset:offset+x.size(0)] (exact mapping)
        return freqs[:, offset : offset + x.size(1)]


def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    cp_group: Group = None,
    high_precision_rope: bool = False,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype paddle.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]
        cp_group (Group): The context parallel group

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    cp_size = get_pg_size(cp_group)
    cp_rank = get_pg_rank(cp_group)

    seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()

    # Handle two different frequency tensor formats:
    # 1. If freqs.size(0) == cu_seqlens[-1]: freqs contains all positions across all sequences
    #    -> Use offset-based mapping for exact positional correspondence
    # 2. Otherwise: freqs contains only max sequence length positions
    #    -> Use traditional mapping without offsets (map first :seqlen part)
    if freqs.dim() >= 1 and freqs.size(1) == cu_seqlens[-1]:
        # CASE 1: Exact mapping with offsets
        # When cp_size==1, every per-segment slice concatenates back to the original freqs.
        # Skip the split+cat and call bshd directly with the original freqs.
        if cp_size == 1:
            return _apply_rotary_pos_emb_bshd(
                t,
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
                high_precision_rope=high_precision_rope,
            )
        # Build packed freqs in one pass, then apply once to the whole packed tensor
        cu_seqlens_list = cu_seqlens.tolist()
        sequence_splits = paddle.split(t, seqlens, axis=1 if t.ndim == 4 else 0)
        freq_slices = []
        for i, x in enumerate(sequence_splits):
            # cu_seqlens[i] is the starting offset of this sequence in the original batch
            seq_start_offset = cu_seqlens_list[i]
            freq_slices.append(
                _get_thd_freqs_on_this_cp_rank(
                    cp_rank, cp_size, x, freqs, seq_start_offset
                )
            )

        freqs_packed = paddle.cat(freq_slices, axis=1)
        # [b,seq,num_heads,head_dim]
        return _apply_rotary_pos_emb_bshd(
            t,
            freqs_packed,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=multi_latent_attention,
            mscale=mscale,
            high_precision_rope=high_precision_rope,
        )
    else:
        # CASE 2: Traditional mapping without offsets
        # Build packed freqs for all sequences using the standard mapping, then apply once
        sequence_splits = paddle.split(t, seqlens, axis=1 if t.ndim == 4 else 0)
        freqs_packed = paddle.cat(
            [
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs)
                for x in sequence_splits
            ],
            axis=1,
        )

        return _apply_rotary_pos_emb_bshd(
            t,
            freqs_packed,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=multi_latent_attention,
            mscale=mscale,
            high_precision_rope=high_precision_rope,
        )


def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    cos: Tensor | None,
    sin: Tensor | None,
    config: TransformerConfig,
    cu_seqlens: Tensor | None = None,
    mscale: float = 1.0,
    cp_group: Group = None,
    position_ids: Tensor | None = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format

    Args:
        t (Tensor): Input tensor
        freqs (Tensor): Rotary positional embedding frequencies
        cos (Tensor | None): Pre-computed cosine values of freqs (used for fused implementation)
        sin (Tensor | None): Pre-computed sine values of freqs (used for fused implementation)
        config (TransformerConfig): Transformer configuration
        cu_seqlens (Tensor | None): Cumulative sequence lengths
        mscale (float): Scaling factor
        cp_group (Group): Context parallel group
    """
    if config.apply_rope_fusion:
        # Paddle fused_rope not support cu_seqlens or cp_group
        if cu_seqlens:
            raise NotImplementedError(
                "cu_seqlens not be supported when using fused_rope"
            )
        else:
            assert isinstance(t, tuple), (
                "The input for fused_rope should be a tuple of tensors"
            )
            return fused_rope(
                *t,
                sin=sin,
                cos=cos,
                rotary_emb_base=config.rope_theta,
                position_ids=position_ids,
                use_neox_rotary_style=config.rotary_interleaved,
                time_major=config.sequence_parallel,
            )

    # use unfused implementation
    if cu_seqlens is None:
        return _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            multi_latent_attention=config.multi_latent_attention,
            mscale=mscale,
            high_precision_rope=config.high_precision_rope,
        )
    else:
        return _apply_rotary_pos_emb_thd(
            t,
            cu_seqlens,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            multi_latent_attention=config.multi_latent_attention,
            mscale=mscale,
            cp_group=cp_group,
            high_precision_rope=config.high_precision_rope,
        )
