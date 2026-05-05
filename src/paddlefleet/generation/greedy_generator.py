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
from typing import Optional

import paddle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class DynamicKVCache:
    """HF-style dynamic KV cache: per-layer tensors grow by concat.

    Implements the ``.update(k_new, v_new, layer_idx) -> (k, v)`` protocol
    expected by :class:`DotProductAttention`.
    """

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

    def __init__(self, fleet_model):
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
        self.cache = DynamicKVCache(num_layers=num_layers)

    @paddle.no_grad()
    def generate(
        self,
        input_ids: paddle.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> paddle.Tensor:
        """Run greedy auto-regressive decoding.

        Args:
            input_ids: Token IDs with shape ``[B, L]``.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: Stop generation when this token is produced (all
                batches must be done).

        Returns:
            Tensor of shape ``[B, L + num_generated]`` containing the
            prompt plus generated tokens.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, L]")
        self.cache.reset()
        self.model.eval()

        bsz, prompt_len = input_ids.shape

        with paddle.amp.auto_cast(True, level="O2", dtype="bfloat16"):
            # ---- Prefill ----
            position_ids = (
                paddle.arange(prompt_len, dtype="int64")
                .unsqueeze(0)
                .expand([bsz, prompt_len])
            )
            logits = self.model({
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": self.cache,
                "use_cache": True,
            })
            next_tok = logits[:, -1].argmax(axis=-1, keepdim=True)
            out_tokens = [next_tok]

            # ---- Decode ----
            done = paddle.zeros([bsz, 1], dtype="bool")
            for _ in range(max_new_tokens - 1):
                cur_len = self.cache.get_seq_len()
                position_ids = paddle.full([bsz, 1], cur_len, dtype="int64")
                logits = self.model({
                    "input_ids": next_tok,
                    "position_ids": position_ids,
                    "past_key_values": self.cache,
                    "use_cache": True,
                })
                next_tok = logits[:, -1].argmax(axis=-1, keepdim=True)
                out_tokens.append(next_tok)
                if eos_token_id is not None:
                    done = done | (next_tok == eos_token_id)
                    if done.all().item():
                        break

        return paddle.concat([input_ids] + out_tokens, axis=1)
