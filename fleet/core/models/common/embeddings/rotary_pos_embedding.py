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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

    from fleet.core.packed_seq_params import PackedSeqParams
    from fleet.core.transformer.transformer_block import TransformerBlock
    from fleet.core.transformer.transformer_config import TransformerConfig

import logging
import math

import paddle
from paddle import Tensor, nn

from fleet.core import parallel_state
from fleet.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank,
)

logger = logging.getLogger(__name__)


__all__ = ["RotaryEmbedding"]


class RotaryEmbedding(nn.Layer):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
        rope_scaling (bool, optional): Apply rope scaling as used in llama 3.x.
        rope_scaling_factor (float, optional): rope scaling factor in llama 3.x. Defaults to 8.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly
            on the GPU. Defaults to False
        cp_group (Group, optional): Process group for context parallel.
            Defaults to None.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float | None = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        use_cpu_initialization: bool = False,
        cp_group: Group | None = None,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        device = "cpu" if use_cpu_initialization else paddle.get_device()
        self.inv_freq = 1.0 / (
            rotary_base
            ** (
                paddle.arange(0, dim, 2, dtype=paddle.float32, device=device)
                / dim
            )
        )

        if rope_scaling:
            self.inv_freq = self._apply_scaling(
                self.inv_freq, factor=rope_scaling_factor
            )

        self.cp_group = (
            cp_group
            if cp_group is not None
            else parallel_state.get_context_parallel_group(
                check_initialized=False
            )
        )

    def _apply_scaling(
        self,
        freqs,
        factor=8,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
    ):
        # This implementation is adapted from:
        # https://github.com/huggingface/transformers/blob/2a5a6ad18aa22e98429bb5ecb880660328030ea0/src/transformers/modeling_rope_utils.py#L303-L343

        factor = factor  # `8` in the original implementation
        low_freq_factor = low_freq_factor  # `1` in the original implementation
        high_freq_factor = (
            high_freq_factor  # `4` in the original implementation
        )
        old_context_len = original_max_position_embeddings  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / freqs
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = paddle.where(
            wavelen > low_freq_wavelen, freqs / factor, freqs
        )
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(
            wavelen > low_freq_wavelen
        )
        inv_freq_llama = paddle.where(
            is_medium_freq, smoothed_inv_freq, inv_freq_llama
        )

        return inv_freq_llama

    def get_freqs_non_repeated(
        self, max_seq_len: int, offset: int = 0
    ) -> Tensor:
        """Generates matrix of frequencies based on positions in the sequence,
        used to create positional encodings"""
        seq = (
            paddle.arange(
                max_seq_len,
                device=self.inv_freq.place,
                dtype=self.inv_freq.dtype,
            )
            + offset
        )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = paddle.outer(seq, self.inv_freq)  # [seq len, dim]

        return freqs

    def get_cos_sin(
        self, max_seq_len: int, offset: int = 0
    ) -> (Tensor, Tensor):
        """Cosine and sine values for RoPE are precomputed for all positions up to the maximum
        sequence length"""
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        cos = paddle.cos(freqs)
        sin = paddle.sin(freqs)
        return cos, sin

    def forward(
        self, max_seq_len: int, offset: int = 0, packed_seq: bool = False
    ) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        if self.inv_freq.place.is_cpu_place():
            # move `inv_freq` to GPU once at the first micro-batch forward pass
            self.inv_freq = self.inv_freq.to(device=paddle.get_device())

        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = paddle.cat((freqs, freqs), axis=-1)
        else:
            emb = paddle.stack(
                (freqs.view(-1, 1), freqs.view(-1, 1)), axis=-1
            ).view(freqs.shape[0], -1)
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        if (
            self.cp_group is not None
            and self.cp_group.size() > 1
            and not packed_seq
        ):
            # slice rotary_pos_emb along sequence dimension and select the partition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)
        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f"{prefix}inv_freq", None)
        return super()._load_from_state_dict(
            state_dict, prefix, *args, **kwargs
        )

    def get_rotary_seq_len(
        self,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> int:
        """Function to get the rotary sequence length.

        Args:
            transformer (TransformerBlock): The transformer block (decoder/encoder) used
                by the model
            transformer_input (Tensor): Input tensor to the transformer
            transformer_config (TransformerConfig): Transformer config used by the model
            packed_seq_params (PackedSeqParams): Packed sequence params

        Returns:
            int: The rotary sequence length
        """

        if packed_seq_params is not None:
            # max_seqlen are the max sequence length in the packed sequence before being divived
            # by the tp and cp size.
            return max(
                packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv
            )
        else:
            if transformer is not None and transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
            else:
                rotary_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len
