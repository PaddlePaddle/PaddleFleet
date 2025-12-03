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
    from paddlefleet.packed_seq_params import PackedSeqParams
    from paddlefleet.transformer.transformer_block import TransformerBlock
    from paddlefleet.transformer.transformer_config import TransformerConfig

import math

import paddle
from paddle import Tensor, nn

from paddlefleet import parallel_state
from paddlefleet.context_parallel_utils import ContextParallelScatterOp

logger = logging.getLogger(__name__)


__all__ = ["RotaryEmbedding"]


class RotaryEmbedding(nn.Layer):
    """Rotary Embedding for language model.

    Args:
        head_dim (int): Projection weights dimension in multi-head attention. Obtained
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
        cp_group (paddle.distributed.communication.group.Group, optional): Process group for context parallel.
            Defaults to None.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float | None = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        cp_group: paddle.distributed.communication.group.Group | None = None,
    ) -> None:
        super().__init__()

        dim = head_dim
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base
            ** (
                paddle.arange(0, dim, 2, dtype=paddle.int64).astype(
                    dtype=paddle.float32
                )
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

        self._cast_to_low_precision = False

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
        seq = paddle.arange(max_seq_len).astype(self.inv_freq.dtype) + offset

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
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = paddle.cat((freqs, freqs), axis=-1)
        else:
            emb = paddle.stack(
                (freqs.reshape((-1, 1)), freqs.reshape((-1, 1))), axis=-1
            ).reshape((freqs.shape[0], -1))
        # emb [1, seq_len, 1, dim]
        emb = emb[None, :, None, :]
        if (
            self.cp_group is not None
            and self.cp_group.world_size > 1
            and not packed_seq
        ):
            # slice rotary_pos_emb along sequence dimension and select the partition of the current
            # CP rank
            emb = ContextParallelScatterOp.apply(emb, axis=1)
        return emb

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
            if (
                transformer_config.sequence_parallel
                and transformer_config.scatter_embedding_sequence_parallel
            ):
                seq_axis = 0
            else:
                seq_axis = 1
            if transformer is not None and transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.shape[seq_axis]
            else:
                rotary_seq_len = transformer_input.shape[seq_axis]

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        # TODO: self.cp_group.world_size --> transformer_config.context_parallel_size
        # rotary_seq_len *= transformer_config.context_parallel_size
        if self.cp_group is not None and self.cp_group.world_size > 1:
            rotary_seq_len *= self.cp_group.world_size

        return rotary_seq_len
