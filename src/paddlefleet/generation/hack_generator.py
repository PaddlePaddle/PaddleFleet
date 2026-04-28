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

"""
TEMPORARY HACK: Greedy inference with KV cache for PaddleFleet.
This file contains the monkey-patch version for quick validation.
It will be refactored in Step 2 into proper implementation.
"""

from __future__ import annotations

import types
from typing import Optional

import paddle
from paddle.nn.functional.flash_attention import flash_attention

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding
from paddlefleet.transformer.transformer_layer import TransformerLayer

# Dynamic KV Cache (will be replaced by StaticKVCache in Step 2)
class DynamicKVCache:
    """HF-style dynamic KV cache: per-layer tensors grow by concat."""

    def __init__(self, num_layers: int):
        self.k: list[Optional[paddle.Tensor]] = [None] * num_layers
        self.v: list[Optional[paddle.Tensor]] = [None] * num_layers

    def get_seq_len(self, layer_idx: int = 0) -> int:
        return 0 if self.k[layer_idx] is None else self.k[layer_idx].shape[1]

    def update(self, k_new: paddle.Tensor, v_new: paddle.Tensor, layer_idx: int):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            self.k[layer_idx] = paddle.concat([self.k[layer_idx], k_new], axis=1)
            self.v[layer_idx] = paddle.concat([self.v[layer_idx], v_new], axis=1)
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self) -> None:
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None


# RoPE patch (may not be needed as PaddleFleet already supports position_ids)
def _patched_get_freqs_non_repeated(
    self,
    max_seq_len: int,
    offset: int = 0,
    position_ids: Optional[paddle.Tensor] = None,
):
    """RotaryEmbedding.get_freqs_non_repeated that honours position_ids."""
    if position_ids is not None:
        seq = position_ids[0] if position_ids.ndim == 2 else position_ids
        seq = seq.astype(self.inv_freq.dtype)
    else:
        seq = paddle.arange(max_seq_len).astype(self.inv_freq.dtype) + offset
    if self.seq_len_interpolation_factor is not None:
        seq *= 1.0 / self.seq_len_interpolation_factor
    return paddle.outer(seq, self.inv_freq)


# Attention patch for KV Cache support
def _patched_core_attention_forward(
    self,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask=None,
    attn_mask_startend_row_indices=None,
    attn_mask_type=None,
    attention_bias=None,
    packed_seq_params=None,
    use_rr_flash_attention: bool = False,
):
    """DotProductAttention.forward with KV-cache support.

    q/k/v arrive as [B, S, num_heads, head_dim] with RoPE already applied.
    New k/v are concatenated to the per-layer cache before computing attention.

    ``causal=True`` is used only during prefill (query length > 1).
    During single-token decode the query must attend to the full cached
    sequence, so ``causal=False``.
    """
    cache = getattr(self, "_kv_cache", None)
    layer_idx = getattr(self, "_layer_idx", None)
    if cache is not None and layer_idx is not None:
        key, value = cache.update(key, value, layer_idx)
    else:
        # No cache, keep original behavior
        key, value = key, value

    is_causal = query.shape[1] > 1

    out, _ = flash_attention(
        query.astype(value.dtype),
        key.astype(value.dtype),
        value,
        dropout=0.0,
        causal=is_causal,
        return_softmax=False,
    )
    bsz, q_len = query.shape[:2]
    return out.reshape([bsz, q_len, -1])


class HackGreedyGenerator:
    """Temporary hack generator for quick validation.

    This class uses DynamicKVCache and monkey-patching to enable
    greedy inference with KV cache on PaddleFleet models.

    In Step 2, this will be refactored into a proper implementation
    with StaticKVCache and native PaddleFleet integration.
    """

    def __init__(self, fleet_model):
        self.model = fleet_model
        self.cache: Optional[DynamicKVCache] = None
        self._patched = False
        self._rope_patched_layers = []
        self._attn_patched_layers = []

    def _ensure_patched(self, kv_cache: Optional[DynamicKVCache] = None) -> None:
        """Apply all necessary patches for inference mode."""
        if self._patched:
            return

        cfg = self.model.config

        # Validate configuration
        if getattr(cfg, "sequence_parallel", False):
            raise ValueError("sequence_parallel must be False for inference")
        if getattr(cfg, "apply_rope_fusion", False):
            raise ValueError(
                "apply_rope_fusion must be False for inference "
                "(set config.apply_rope_fusion=False before building the model)"
            )

        num_layers = 0
        for sub in self.model.run_function:
            # Patch GPTEmbedding RotaryEmbedding to support position_ids
            if isinstance(sub, GPTEmbedding):
                if sub.rotary_pos_emb is not None:
                    sub.rotary_pos_emb.get_freqs_non_repeated = types.MethodType(
                        _patched_get_freqs_non_repeated, sub.rotary_pos_emb
                    )
                    self._rope_patched_layers.append(num_layers)

            # Patch TransformerLayer core_attention for KV cache
            elif isinstance(sub, TransformerLayer):
                if hasattr(sub, "self_attn"):
                    core = sub.self_attn.core_attention
                    core._kv_cache = kv_cache
                    core._layer_idx = num_layers
                    core.forward = types.MethodType(
                        _patched_core_attention_forward, core
                    )
                    self._attn_patched_layers.append(num_layers)

                num_layers += 1

        if num_layers == 0:
            raise RuntimeError("No valid layers found in model.run_function")

        self._patched = True
        self.cache = kv_cache

    @paddle.no_grad()
    def generate(
        self,
        input_ids: paddle.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> paddle.Tensor:
        """Generate text using greedy decoding with KV cache."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, L]")

        self._ensure_patched(DynamicKVCache(num_layers=self.model.config.num_hidden_layers))
        self.cache.reset()
        self.model.eval()

        bsz, prompt_len = input_ids.shape

        with paddle.amp.auto_cast(True, level="O2", dtype="bfloat16"):
            # Prefill phase
            position_ids = (
                paddle.arange(prompt_len, dtype="int64")
                .unsqueeze(0)
                .expand([bsz, prompt_len])
            )
            logits = self.model(
                {"input_ids": input_ids, "position_ids": position_ids}
            )
            next_tok = logits[:, -1].argmax(axis=-1, keepdim=True)
            out_tokens = [next_tok]

            # Decode phase
            done = paddle.zeros([bsz, 1], dtype="bool")
            for _ in range(max_new_tokens - 1):
                cur_len = self.cache.get_seq_len()
                position_ids = paddle.full([bsz, 1], cur_len, dtype="int64")

                logits = self.model(
                    {"input_ids": next_tok, "position_ids": position_ids}
                )
                next_tok = logits[:, -1].argmax(axis=-1, keepdim=True)
                out_tokens.append(next_tok)

                if eos_token_id is not None:
                    done = done | (next_tok == eos_token_id)
                    if done.all().item():
                        break

        return paddle.concat([input_ids] + out_tokens, axis=1)