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

"""Greedy inference for Fleet models using the native KV cache path.

PaddleFleet already wires KV cache through the entire stack:

* ``GPTEmbedding.forward`` sets ``_kv_layer_counter: 0`` in the output dict.
* ``TransformerLayer.forward`` reads / increments the counter and passes
  ``past_key_values``, ``layer_idx``, ``use_cache`` through to the attention
  layers.
* ``DotProductAttention.forward`` calls ``past_key_values.update(k, v,
  layer_idx)`` and switches causal masking based on query length.

This module provides a :class:`DynamicKVCache` that satisfies the
``.update(k, v, layer_idx) -> (k, v)`` protocol and a :class:`GreedyGenerator`
that drives the prefill / decode loop.

Usage::

    from paddlefleet.generation import GreedyGenerator

    gen = GreedyGenerator(model)
    out_ids = gen.generate(input_ids, max_new_tokens=128,
                           eos_token_id=tok.eos_token_id)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import paddle

from paddlefleet.transformer.utils import is_layer_window_attention

if TYPE_CHECKING:
    from paddlefleet.models.gpt.gpt_model import GPTModel

logger = logging.getLogger(__name__)
_DEBUG = os.environ.get("GREEDY_DEBUG", "0") == "1"


def _apply_repetition_penalty(
    logits: paddle.Tensor, input_ids: paddle.Tensor, penalty: float
) -> paddle.Tensor:
    """Apply repetition penalty to logits.

    Tokens with positive logits are divided by penalty,
    tokens with negative logits are multiplied by penalty.
    """
    if penalty == 1.0:
        return logits

    batch_size, seq_len = input_ids.shape
    vocab_size = logits.shape[-1]

    # Create mask for tokens that appeared in input_ids using scatter
    # This is more efficient than the loop version
    token_mask = paddle.zeros([batch_size, vocab_size], dtype="float32")

    # Flatten input_ids and create batch indices
    flat_input_ids = input_ids.reshape([-1])  # [batch_size * seq_len]
    batch_indices = paddle.arange(batch_size, dtype="int64").unsqueeze(-1)
    batch_indices = batch_indices.expand([batch_size, seq_len]).reshape(
        [-1]
    )  # [batch_size * seq_len]

    # Create indices for scatter
    scatter_indices = paddle.stack(
        [batch_indices, flat_input_ids], axis=-1
    )  # [batch_size * seq_len, 2]

    # Scatter 1.0 to mark appeared tokens
    token_mask = paddle.scatter_nd(
        scatter_indices,
        paddle.ones([batch_size * seq_len], dtype="float32"),
        [batch_size, vocab_size],
    )

    # Apply penalty: divide positive, multiply negative
    mask = token_mask > 0
    logits = paddle.where(
        mask,
        paddle.where(logits > 0, logits / penalty, logits * penalty),
        logits,
    )
    return logits


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class DynamicKVCache:
    """HF-style dynamic KV cache: per-layer tensors grow by concat.

    Implements the ``.update(k_new, v_new, layer_idx) -> (k, v)`` protocol
    expected by :class:`DotProductAttention`.
    """

    def __init__(
        self,
        num_layers: int,
        swa_layers: list[bool] | None = None,
        window_size: int | None = None,
    ):
        self.k: list[paddle.Tensor | None] = [None] * num_layers
        self.v: list[paddle.Tensor | None] = [None] * num_layers
        self.swa_layers = swa_layers or [False] * num_layers
        self.window_size = window_size
        # Track absolute sequence length independently of truncated cache
        self._seq_len: list[int] = [0] * num_layers

    def get_seq_len(self, layer_idx: int = 0) -> int:
        if self._seq_len[layer_idx] > 0:
            return self._seq_len[layer_idx]
        # Fallback: find first non-zero layer
        for s in self._seq_len:
            if s > 0:
                return s
        return 0

    def update(
        self, k_new: paddle.Tensor, v_new: paddle.Tensor, layer_idx: int
    ):
        if self.k[layer_idx] is None:
            self.k[layer_idx] = k_new
            self.v[layer_idx] = v_new
        else:
            self.k[layer_idx] = paddle.concat(
                [self.k[layer_idx], k_new], axis=1
            )
            self.v[layer_idx] = paddle.concat(
                [self.v[layer_idx], v_new], axis=1
            )
        # Update absolute sequence length before truncation
        self._seq_len[layer_idx] = self._seq_len[layer_idx] + k_new.shape[1]
        # Return full K/V for current attention computation
        full_k, full_v = self.k[layer_idx], self.v[layer_idx]
        # Truncate SWA layers in cache for subsequent decode steps
        if self.window_size and self.swa_layers[layer_idx]:
            self.k[layer_idx] = self.k[layer_idx][:, -self.window_size :]
            self.v[layer_idx] = self.v[layer_idx][:, -self.window_size :]
        return full_k, full_v

    def reset(self) -> None:
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None
            self._seq_len[i] = 0


# ---------------------------------------------------------------------------
# Greedy generator
# ---------------------------------------------------------------------------


class GreedyGenerator:
    """Greedy decode on top of a FleetGPTModel using the native KV cache path.

    No monkey-patching is needed — the model's own forward pass already
    supports KV cache via the ``past_key_values`` / ``use_cache`` mechanism.

    Usage::

        model = Qwen3MoeForCausalLM.from_pretrained(model_dir, config=config)
        gen = GreedyGenerator(model)
        out = gen.generate(input_ids, max_new_tokens=128,
                           eos_token_id=tok.eos_token_id)
    """

    def __init__(self, fleet_model: GPTModel):
        cfg = fleet_model.config

        if getattr(cfg, "sequence_parallel", False):
            raise ValueError(
                "sequence_parallel must be False for inference with KV cache. "
                "Set config.sequence_parallel = False before building the model."
            )
        if getattr(cfg, "apply_rope_fusion", False):
            logger.warning(
                "apply_rope_fusion=True may cause issues with KV cache "
                "inference. If outputs are incorrect, set "
                "config.apply_rope_fusion = False."
            )
        if getattr(cfg, "recompute_granularity", None) == "full":
            logger.warning(
                "recompute_granularity='full' drops KV cache kwargs. "
                "Make sure model.eval() is called before generate()."
            )

        self.model = fleet_model
        num_layers = cfg.num_hidden_layers
        # Account for empty layers in head/tail that offset layer_number
        num_empty_layers_add_in_head = getattr(
            cfg, "num_empty_layers_add_in_head", 0
        )
        num_empty_layers_add_in_tail = getattr(
            cfg, "num_empty_layers_add_in_tail", 0
        )
        total_layers = (
            num_layers
            + num_empty_layers_add_in_head
            + num_empty_layers_add_in_tail
        )

        # Determine which layers use SWA for KV cache truncation
        sliding_window = getattr(cfg, "sliding_window", None)
        window_attn_skip_freq = getattr(cfg, "window_attn_skip_freq", None)
        swa_layers = []
        for i in range(total_layers):
            real_i = i - num_empty_layers_add_in_head
            if real_i < 0 or real_i >= num_layers:
                swa_layers.append(False)
            else:
                swa_layers.append(
                    is_layer_window_attention(
                        sliding_window, window_attn_skip_freq, real_i
                    )
                )
        self.window_size = (
            sliding_window[0]
            if sliding_window and sliding_window[0] > 0
            else None
        )
        # head_wise_swa_ratio > 0 means some heads in SWA layers use full
        # attention; per-layer KV truncation would break those heads.
        head_wise_swa_ratio = getattr(cfg, "head_wise_swa_ratio", 0.0)
        if head_wise_swa_ratio > 0 and head_wise_swa_ratio < 1.0:
            raise ValueError(
                f"head_wise_swa_ratio={head_wise_swa_ratio}: KV cache truncation "
                "is unsupported because it would break full-attention heads in "
                "SWA layers."
            )
        self.cache = DynamicKVCache(
            num_layers=total_layers,
            swa_layers=swa_layers,
            window_size=self.window_size,
        )

    @paddle.no_grad()
    def generate(
        self,
        input_ids: paddle.Tensor,
        max_new_tokens: int,
        eos_token_id: int | list[int] | None = None,
        repetition_penalty: float = 1.0,
    ) -> paddle.Tensor:
        """Run greedy auto-regressive decoding.

        Args:
            input_ids: Token IDs with shape ``[B, L]``.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: Stop generation when this token is produced (all
                batches must be done).
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty,
                >1.0 = discourage repetition). Default: 1.0.

        Returns:
            Tensor of shape ``[B, L + num_generated]`` containing the
            prompt plus generated tokens.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, L]")
        self.cache.reset()
        self.model.eval()

        bsz, prompt_len = input_ids.shape
        generated = input_ids.clone()

        _r = (
            paddle.distributed.get_rank()
            if paddle.distributed.is_initialized()
            else 0
        )

        with paddle.amp.auto_cast(True, level="O2", dtype="bfloat16"):
            # ---- Prefill ----
            position_ids = (
                paddle.arange(prompt_len, dtype="int64")
                .unsqueeze(0)
                .expand([bsz, prompt_len])
            )
            if _DEBUG and _r == 0:
                logger.info(
                    "[fwd-debug][prefill] input_ids=%s position_ids=%s",
                    input_ids.tolist(),
                    position_ids.tolist(),
                )

            if self.cache.window_size and any(self.cache.swa_layers):
                prefill_startend = paddle.full(
                    [bsz, 1, prompt_len, 1], prompt_len, dtype="int32"
                )
            else:
                prefill_startend = None
            logits = self.model(
                {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "past_key_values": self.cache,
                    "use_cache": True,
                    "attn_mask_startend_row_indices": prefill_startend,
                }
            )
            if _DEBUG and _r == 0:
                _last = logits[:, -1].cast("float32")
                logger.info(
                    "[logits-debug][prefill] shape=%s dtype=%s",
                    list(logits.shape),
                    logits.dtype,
                )
                logger.info(
                    "[logits-debug][prefill] min=%.4f max=%.4f mean=%.4f",
                    float(_last.min()),
                    float(_last.max()),
                    float(_last.mean()),
                )
                _topk_val, _topk_idx = paddle.topk(_last[0], k=10)
                logger.info(
                    "[logits-debug][prefill] top-10 ids=%s vals=%s",
                    _topk_idx.tolist(),
                    [round(v, 3) for v in _topk_val.tolist()],
                )
                _kv_lens = [
                    self.cache._seq_len[i]
                    for i in range(min(3, len(self.cache._seq_len)))
                    if self.cache._seq_len[i] > 0
                ]
                logger.info(
                    "[kv-debug][prefill] cache seq_lens (first 3 non-zero layers): %s",
                    _kv_lens,
                )

            # Apply repetition penalty to prefill output
            logits = _apply_repetition_penalty(
                logits, generated, repetition_penalty
            )
            next_tok = logits[:, -1].argmax(axis=-1, keepdim=True)
            generated = paddle.concat([generated, next_tok], axis=1)

            # ---- Decode ----
            done = paddle.zeros([bsz, 1], dtype="bool")
            for step in range(max_new_tokens - 1):
                cur_len = self.cache.get_seq_len()
                position_ids = paddle.full([bsz, 1], cur_len, dtype="int64")
                if _DEBUG and _r == 0 and step < 3:
                    logger.info(
                        "[fwd-debug][decode step=%d] next_tok=%s position_ids=%s cache_seq_len=%d",
                        step,
                        next_tok.tolist(),
                        position_ids.tolist(),
                        cur_len,
                    )
                logits = self.model(
                    {
                        "input_ids": next_tok,
                        "position_ids": position_ids,
                        "past_key_values": self.cache,
                        "use_cache": True,
                    }
                )
                if _DEBUG and _r == 0 and step < 3:
                    _d = logits[:, -1].cast("float32")
                    _tv, _ti = paddle.topk(_d[0], k=5)
                    logger.info(
                        "[logits-debug][decode step=%d] top-5 ids=%s vals=%s",
                        step,
                        _ti.tolist(),
                        [round(v, 3) for v in _tv.tolist()],
                    )
                # Apply repetition penalty
                logits = _apply_repetition_penalty(
                    logits, generated, repetition_penalty
                )
                next_tok = logits[:, -1].argmax(axis=-1, keepdim=True)
                generated = paddle.concat([generated, next_tok], axis=1)
                if eos_token_id is not None:
                    if isinstance(eos_token_id, list):
                        # eos_token_id may be nested list like [[t1,t2],[t3]]
                        # extract single-token stop ids for fast eos check
                        flat_ids = [
                            ids[0]
                            if isinstance(ids, list) and len(ids) == 1
                            else ids
                            for ids in eos_token_id
                            if not isinstance(ids, list) or len(ids) == 1
                        ]
                        if flat_ids:
                            eos_tensor = paddle.to_tensor(
                                flat_ids, dtype=next_tok.dtype
                            ).reshape([1, 1, -1])
                            hit = (next_tok.unsqueeze(-1) == eos_tensor).any(
                                axis=-1
                            )
                            done = done | hit
                    else:
                        done = done | (next_tok == eos_token_id)
                    if done.all().item():
                        break

        return generated
