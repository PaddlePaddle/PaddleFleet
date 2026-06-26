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
"""Gemma4 layer specs, DualRotaryEmbedding, Embedding, and OutputLayer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
from paddle import nn

from paddlefleet.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


# ============================================================
# Gemma4 Embedding (with sqrt(hidden_size) scaling)
# ============================================================


class Gemma4Embedding(LanguageModelEmbedding):
    """Embedding scaled by sqrt(hidden_size), standard for all Gemma models."""

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        embeddings = super().forward(input_ids, position_ids, tokentype_ids)
        return embeddings * (self.config.hidden_size**0.5)


# ============================================================
# Gemma4 Dual Rotary Embedding
# ============================================================


class Gemma4ProportionalRotaryEmbedding(nn.Layer):
    """Proportional RoPE for global layers (aligned with HF _compute_proportional_rope_parameters).

    Unlike RotaryEmbedding with rotary_percent<1.0 (which truncates dim and uses partial
    rotation via t/t_pass split), proportional RoPE zero-pads inv_freq so that cos/sin
    have the same dimension as head_dim. This means rotate_half operates on the full
    head_dim, and non-rotated dimensions naturally have cos=1, sin=0.

    HF implementation (modeling_rope_utils.py:187):
        rope_angles = int(partial_rotary_factor * head_dim // 2)
        inv_freq_rotated = 1.0 / (base ** (arange(0, 2*rope_angles, 2) / head_dim))
        nope_angles = head_dim // 2 - rope_angles
        inv_freq = cat(inv_freq_rotated, zeros(nope_angles))
    """

    def __init__(
        self,
        head_dim: int,
        rotary_base: float,
        partial_rotary_factor: float = 0.25,
    ):
        super().__init__()
        self.head_dim = head_dim
        rope_angles = int(partial_rotary_factor * head_dim // 2)

        # Rotated part: arange(0, 2*rope_angles, 2) / head_dim
        inv_freq_rotated = 1.0 / (
            rotary_base
            ** (
                paddle.arange(0, 2 * rope_angles, 2, dtype=paddle.int64).cast(
                    paddle.float32
                )
                / head_dim
            )
        )

        # Non-rotated part: zeros
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = paddle.concat(
                [
                    inv_freq_rotated,
                    paddle.zeros([nope_angles], dtype=paddle.float32),
                ],
                axis=0,
            )
        else:
            inv_freq = inv_freq_rotated

        self.inv_freq = inv_freq  # [head_dim // 2]

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        position_ids=None,
    ):
        """Compute angle freqs with full head_dim (zero-padded proportional RoPE).

        Returns raw angles in the same format as RotaryEmbedding.forward():
        [1, seq_len, 1, head_dim] where head_dim = 2 * len(inv_freq).

        Zero-padded inv_freq ensures rot_dim == head_dim in
        _apply_rotary_pos_emb_bshd, so no t/t_pass split occurs.
        Non-rotated dimensions naturally have cos=1, sin=0 (identity rotation).
        """
        if position_ids is not None:
            # position_ids: [batch, seq_len]
            inv_freq_expanded = self.inv_freq[None, :, None]  # [1, dim/2, 1]
            position_ids_expanded = position_ids[:, None, :].cast(
                paddle.float32
            )  # [batch, 1, seq_len]
            freqs = paddle.matmul(
                inv_freq_expanded, position_ids_expanded
            ).transpose([0, 2, 1])  # [batch, seq_len, dim/2]
        else:
            t = paddle.arange(
                offset, max_seq_len + offset, dtype=self.inv_freq.dtype
            )
            freqs = paddle.outer(t, self.inv_freq)  # [seq_len, dim/2]
            freqs = freqs[None, :, :]  # [1, seq_len, dim/2]

        # Duplicate freqs to match rotate_half pairing (same as RotaryEmbedding)
        emb = paddle.cat([freqs, freqs], axis=-1)  # [..., head_dim]

        # Add head dimension: [batch_or_1, seq_len, 1, head_dim]
        if emb.ndim == 3:
            emb = emb[:, :, None, :]

        return emb


class DualRoPEOutput:
    """Tuple-like container for (rope_sliding, rope_global) that supports .clone().

    TransformerLayer.forward() calls rotary_pos_emb.clone() which fails on a
    plain tuple. This wrapper delegates clone() to each element and supports
    indexing/len so Gemma4SelfAttention can do rotary_pos_emb[0] / [1].
    """

    def __init__(self, local_emb, global_emb):
        self.local_emb = local_emb
        self.global_emb = global_emb

    def __getitem__(self, idx):
        if idx == 0:
            return self.local_emb
        elif idx == 1:
            return self.global_emb
        raise IndexError(f"DualRoPEOutput index out of range: {idx}")

    def __len__(self):
        return 2

    def clone(self):
        return DualRoPEOutput(self.local_emb.clone(), self.global_emb.clone())


class Gemma4DualRotaryEmbedding(nn.Layer):
    """Dual RoPE: sliding (theta=10000, full rotation) + global (theta=1M, proportional).

    Returns a DualRoPEOutput (rope_sliding, rope_global) for layer-level selection.
    """

    def __init__(self, config):
        super().__init__()
        kv_channels = getattr(config, "kv_channels", 256)
        global_head_dim = getattr(config, "global_head_dim", 512)
        sliding_rope_base = getattr(config, "sliding_window_rope_base", 10000)
        global_rope_base = getattr(config, "full_attention_rope_base", 1000000)
        global_rotary_percent = getattr(config, "global_rotary_percent", 0.25)

        self.rope_local = RotaryEmbedding(
            head_dim=kv_channels,
            rotary_percent=1.0,
            rotary_base=sliding_rope_base,
        )
        # Use proportional RoPE for global layers (zero-padded inv_freq matching HF)
        self.rope_global = Gemma4ProportionalRotaryEmbedding(
            head_dim=global_head_dim,
            rotary_base=global_rope_base,
            partial_rotary_factor=global_rotary_percent,
        )

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        position_ids=None,
    ):
        local_emb = self.rope_local(
            max_seq_len,
            offset,
            packed_seq=packed_seq,
            position_ids=position_ids,
        )
        global_emb = self.rope_global(
            max_seq_len,
            offset,
            packed_seq=packed_seq,
            position_ids=position_ids,
        )
        return DualRoPEOutput(local_emb, global_emb)

    def get_rotary_seq_len(
        self, transformer_input, transformer_config, packed_seq_params=None
    ):
        """Delegate to rope_local (shares the same sequence length logic)."""
        return self.rope_local.get_rotary_seq_len(
            transformer_input, transformer_config, packed_seq_params
        )


# ============================================================
# Gemma4 Output Layer (Logit Softcapping)
# ============================================================


class Gemma4OutputLayer(nn.Layer):
    """Wrapper for logit softcapping: tanh(x/cap) * cap."""

    def __init__(self, original_output_layer, softcap: float = 30.0):
        super().__init__()
        self.linear = original_output_layer
        self.softcap = softcap

    def forward(self, hidden_states):
        logits = self.linear(hidden_states)
        if isinstance(logits, tuple):
            logits, bias = logits
            logits = paddle.tanh(logits / self.softcap) * self.softcap
            return logits, bias
        return paddle.tanh(logits / self.softcap) * self.softcap


# ============================================================
# Layer Spec Factory (thin wrapper for backward compatibility)
# ============================================================


def get_gemma4_decoder_layers_spec(config: TransformerConfig) -> list:
    """Generate layer specs for all Gemma4 layers via standard GPT path."""
    from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from paddlefleet.transformer.transformer_layer import Gemma4TransformerLayer

    config.specific_layer = Gemma4TransformerLayer
    num_layers = getattr(config, "num_layers", 30)
    return [
        get_gpt_layer_local_spec(
            config=config,
            num_experts=None,
            use_qk_norm=True,
            normalization=getattr(config, "normalization", "RMSNorm"),
            layer_number=i + 1,
            attention_layer_type="gemma4",
        )
        for i in range(num_layers)
    ]
