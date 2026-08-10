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

from paddlefleet.transformer.utils import (
    get_sliding_window_left_size,
    is_layer_window_attention,
)

if TYPE_CHECKING:
    from paddlefleet.models.gpt.gpt_model import GPTModel

logger = logging.getLogger(__name__)
_DEBUG = os.environ.get("GREEDY_DEBUG", "0") == "1"


def _sample_next_token(
    logits: paddle.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> paddle.Tensor:
    """Sample next token from last-position logits ``[B, vocab]``.

    Falls back to greedy (argmax) when temperature==0 or top_k==1.

    Args:
        logits: Shape ``[B, vocab_size]``, last-position logits.
        temperature: Softmax temperature. 0 or negative means greedy.
        top_k: Keep only top-k tokens before sampling. 0 = disabled.
        top_p: Nucleus sampling threshold in (0, 1]. 0.0 = disabled.

    Returns:
        Shape ``[B, 1]`` sampled token ids.
    """
    # Greedy shortcut
    if temperature <= 0.0 or top_k == 1:
        return logits.argmax(axis=-1, keepdim=True)

    logits = logits.cast("float32")

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        vocab_size = logits.shape[-1]
        k = min(top_k, vocab_size)
        # topk returns (values, indices); keep the k-th value as threshold
        top_k_vals = paddle.topk(logits, k=k, axis=-1)[0]  # [B, k]
        threshold = top_k_vals[:, -1].unsqueeze(-1)  # [B, 1]
        logits = paddle.where(
            logits < threshold, paddle.full_like(logits, float("-inf")), logits
        )

    # Top-p (nucleus) filtering
    if top_p > 0.0 and top_p < 1.0:
        sorted_indices = paddle.argsort(logits, axis=-1, descending=True)
        sorted_logits = paddle.take_along_axis(logits, sorted_indices, axis=1)
        cumulative_probs = paddle.nn.functional.softmax(
            sorted_logits, axis=-1
        ).cumsum(axis=-1)
        # Remove tokens whose cumulative prob exceeds top_p (shift by 1 to keep first over-threshold)
        sorted_mask = (
            cumulative_probs
            - paddle.nn.functional.softmax(sorted_logits, axis=-1)
        ) >= top_p
        sorted_logits = paddle.where(
            sorted_mask,
            paddle.full_like(sorted_logits, float("-inf")),
            sorted_logits,
        )
        # Scatter back to original order
        original_order = paddle.argsort(sorted_indices, axis=-1)
        logits = paddle.take_along_axis(sorted_logits, original_order, axis=1)

    probs = paddle.nn.functional.softmax(logits, axis=-1)
    next_tok = paddle.multinomial(probs, num_samples=1)  # [B, 1]
    return next_tok


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

    HySparse adds a second, cross-layer slot: a full-attention layer publishes
    its compressed KV latent (``shared_key``) for the downstream SWA (MQA)
    layers' block-sparse branch. During training that tensor is threaded through
    ``dict_args``; during incremental decode only the new token's slice is
    computed, so it is accumulated here via :meth:`update_shared`.
    """

    def __init__(
        self,
        num_layers: int,
        swa_layers: list[bool] | None = None,
        window_size: int | None = None,
    ):
        self.k: list[paddle.Tensor | None] = [None] * num_layers
        self.v: list[paddle.Tensor | None] = [None] * num_layers
        # HySparse shared (compressed) KV latent produced by full layers.
        self.shared_k: list[paddle.Tensor | None] = [None] * num_layers
        self.swa_layers = swa_layers or [False] * num_layers
        self.window_size = window_size
        # Track absolute sequence length independently of truncated cache
        self._seq_len: list[int] = [0] * num_layers

    def has_layer_cache(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` already cached something (i.e. we are decoding).

        Attention layers use this to distinguish the first (prefill) pass from
        the incremental decode steps that follow, instead of relying on the
        query length -- a one-token prompt is still a prefill.
        """
        return self.k[layer_idx] is not None

    def get_layer_kv(
        self, layer_idx: int
    ) -> tuple[paddle.Tensor | None, paddle.Tensor | None]:
        """Cached ``(k, v)`` of ``layer_idx`` (window-truncated for SWA layers).

        HySparse full layers read their own cached keys back to score blocks for
        the current decode token.
        """
        return self.k[layer_idx], self.v[layer_idx]

    def update_shared(
        self, shared_key: paddle.Tensor, layer_idx: int
    ) -> paddle.Tensor:
        """Append a full layer's shared KV latent and return the whole history.

        ``shared_key`` is ``[B, S, 1, kv_lora_rank + qk_rope_head_dim]``; it is
        never window-truncated because the block-sparse branch gathers blocks
        from the entire document.
        """
        if self.shared_k[layer_idx] is None:
            self.shared_k[layer_idx] = shared_key
        else:
            self.shared_k[layer_idx] = paddle.concat(
                [self.shared_k[layer_idx], shared_key], axis=1
            )
        return self.shared_k[layer_idx]

    def get_shared(self, layer_idx: int) -> paddle.Tensor | None:
        """Full shared KV latent published by full layer ``layer_idx``."""
        return self.shared_k[layer_idx]

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
            self.shared_k[i] = None
            self._seq_len[i] = 0


# ---------------------------------------------------------------------------
# Greedy generator
# ---------------------------------------------------------------------------


def _uses_dsv4_hybrid_attention(cfg) -> bool:
    """Return True when the model uses DSv4 Hybrid (CSA/HCA) attention.

    Detected from the config: either the attention-variant selector or the
    presence of per-layer compression ratios drives the CSA/HCA code path.
    """
    if getattr(cfg, "experimental_attention_variant", None) == "dsv4_hybrid":
        return True
    ratios = getattr(cfg, "csa_compress_ratios", None)
    return bool(ratios)


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
        # Extract the left (past) window size, supporting both int and tuple
        # forms; a non-positive size (e.g. -1 "infinite window") disables
        # KV-cache truncation.
        self.window_size = None
        if sliding_window:
            left_window = get_sliding_window_left_size(sliding_window)
            if left_window > 0:
                self.window_size = left_window
        # head_wise_swa_ratio > 0 means some heads in SWA layers use full
        # attention; per-layer KV truncation would break those heads.
        head_wise_swa_ratio = getattr(cfg, "head_wise_swa_ratio", 0.0)
        if head_wise_swa_ratio > 0 and head_wise_swa_ratio < 1.0:
            raise ValueError(
                f"head_wise_swa_ratio={head_wise_swa_ratio}: KV cache truncation "
                "is unsupported because it would break full-attention heads in "
                "SWA layers."
            )
        # DSv4 Hybrid (CSA/HCA) attention needs the richer per-layer decode
        # state (raw + compressed KV, group buffers, indexer keys). Its cache
        # also satisfies the standard ``update(k, v, layer_idx)`` protocol, so
        # it can serve models that interleave CSA/HCA and standard layers.
        if _uses_dsv4_hybrid_attention(cfg):
            from paddlefleet.generation.csa_cache import CSADynamicCache

            self.cache = CSADynamicCache(num_layers=total_layers)
        else:
            self.cache = DynamicKVCache(
                num_layers=total_layers,
                swa_layers=swa_layers,
                window_size=self.window_size,
            )

    @paddle.no_grad()
    def _generate_no_cache(
        self,
        generated: paddle.Tensor,
        max_new_tokens: int,
        eos_token_id: int | list[int] | None,
        repetition_penalty: float,
        temperature: float,
        top_k: int,
        top_p: float,
        return_log_probs: bool,
        log_probs_per_batch: list[list[float]] | None,
    ) -> paddle.Tensor | tuple[paddle.Tensor, list[list[float]]]:
        """No-KVCache generation: each step runs a full prefill."""
        bsz = generated.shape[0]
        _r = (
            paddle.distributed.get_rank()
            if paddle.distributed.is_initialized()
            else 0
        )
        with paddle.amp.auto_cast(True, level="O2", dtype="bfloat16"):
            done = paddle.zeros([bsz, 1], dtype="bool")
            for step in range(max_new_tokens):
                cur_len = generated.shape[1]
                position_ids = (
                    paddle.arange(cur_len, dtype="int64")
                    .unsqueeze(0)
                    .expand([bsz, cur_len])
                )
                # Align log labels with the KV-cache path so the two runs can
                # be diffed directly: no_cache step 0 == cached prefill,
                # no_cache step s>0 == cached decode step s-1.
                _tag = "prefill" if step == 0 else f"decode step={step - 1}"
                _log_this_step = _DEBUG and _r == 0 and step < 4
                if _log_this_step:
                    logger.info(
                        "[fwd-debug][no_cache %s] input_ids=%s position_ids=%s cur_len=%d",
                        _tag,
                        generated.tolist(),
                        position_ids.tolist(),
                        cur_len,
                    )
                if self.cache.window_size and any(self.cache.swa_layers):
                    prefill_startend = paddle.full(
                        [bsz, 1, cur_len, 1], cur_len, dtype="int32"
                    )
                else:
                    prefill_startend = None
                logits = self.model(
                    {
                        "input_ids": generated,
                        "position_ids": position_ids,
                        "use_cache": False,
                        "attn_mask_startend_row_indices": prefill_startend,
                    }
                )
                if _log_this_step:
                    _last = logits[:, -1].cast("float32")
                    logger.info(
                        "[logits-debug][no_cache %s] shape=%s dtype=%s",
                        _tag,
                        list(logits.shape),
                        logits.dtype,
                    )
                    logger.info(
                        "[logits-debug][no_cache %s] min=%.4f max=%.4f mean=%.4f",
                        _tag,
                        float(_last.min()),
                        float(_last.max()),
                        float(_last.mean()),
                    )
                    _topk_val, _topk_idx = paddle.topk(
                        _last[0], min(5, _last.shape[-1])
                    )
                    logger.info(
                        "[logits-debug][no_cache %s] top-k ids=%s vals=%s",
                        _tag,
                        _topk_idx.tolist(),
                        [round(v, 3) for v in _topk_val.tolist()],
                    )
                logits = _apply_repetition_penalty(
                    logits, generated, repetition_penalty
                )
                last_logits = logits[:, -1]  # [B, vocab]
                next_tok = _sample_next_token(
                    last_logits, temperature, top_k, top_p
                )
                if return_log_probs and log_probs_per_batch is not None:
                    step_log_probs = paddle.nn.functional.log_softmax(
                        last_logits.cast("float32"), axis=-1
                    )  # [B, vocab]
                    for b in range(bsz):
                        if not bool(done[b, 0].item()):
                            tok_id = int(next_tok[b, 0].item())
                            log_probs_per_batch[b].append(
                                float(step_log_probs[b, tok_id].item())
                            )
                generated = paddle.concat([generated, next_tok], axis=1)
                if eos_token_id is not None:
                    if isinstance(eos_token_id, list):
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
        if return_log_probs:
            return generated, log_probs_per_batch
        return generated

    @paddle.no_grad()
    def generate(
        self,
        input_ids: paddle.Tensor,
        max_new_tokens: int,
        eos_token_id: int | list[int] | None = None,
        repetition_penalty: float = 1.0,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 0.0,
        return_log_probs: bool = False,
        no_cache: bool = False,
    ) -> paddle.Tensor | tuple[paddle.Tensor, list[list[float]]]:
        """Run auto-regressive decoding with optional sampling.

        Args:
            input_ids: Token IDs with shape ``[B, L]``.
            max_new_tokens: Maximum number of new tokens to generate.
            eos_token_id: Stop generation when this token is produced (all
                batches must be done).
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty,
                >1.0 = discourage repetition). Default: 1.0.
            temperature: Softmax temperature for sampling. ``0`` (default) or
                negative means greedy (argmax). Positive values enable sampling.
            top_k: If > 0, only sample from the top-k most probable tokens.
                Ignored when greedy. Default: 0 (disabled).
            top_p: Nucleus sampling threshold in ``(0, 1]``. If 0.0 (default),
                nucleus filtering is disabled. Cannot be combined with top_k > 0
                at the same time as top_k; whichever is set takes effect first
                (top_k then top_p are applied sequentially when both are set).
            return_log_probs: If True, also return the log-probabilities of
                each generated token. Default: False.

                **Semantics**: the log-prob at each step is computed as
                ``log_softmax`` over the logits *after* repetition-penalty
                but *before* temperature scaling / top-k / top-p filtering.
                This is the model's raw (penalised) distribution, not the
                actual sampling distribution used to draw the token.

        Returns:
            If ``return_log_probs=False``: Tensor of shape
            ``[B, L + num_generated]`` containing the prompt plus generated
            tokens.

            If ``return_log_probs=True``: A tuple
            ``(generated, output_log_probs)`` where ``generated`` is as above
            and ``output_log_probs`` is a list of length ``B``, each element
            being a list of per-step log-probs for the generated tokens only
            (length = actual number of generated tokens for that sequence).
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B, L]")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens should be >= 0")
        self.model.eval()

        bsz, prompt_len = input_ids.shape
        generated = input_ids.clone()

        # Per-batch list to accumulate generated-token log-probs
        log_probs_per_batch: list[list[float]] | None = (
            [[] for _ in range(bsz)] if return_log_probs else None
        )

        _r = (
            paddle.distributed.get_rank()
            if paddle.distributed.is_initialized()
            else 0
        )
        if no_cache:
            return self._generate_no_cache(
                generated,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=return_log_probs,
                log_probs_per_batch=log_probs_per_batch,
            )

        self.cache.reset()

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
                _topk_val, _topk_idx = paddle.topk(
                    _last[0], min(5, _last.shape[-1])
                )
                logger.info(
                    "[logits-debug][prefill] top-k ids=%s vals=%s",
                    _topk_idx.tolist(),
                    [round(v, 3) for v in _topk_val.tolist()],
                )
                _kv_lens = [
                    self.cache.get_seq_len(i)
                    for i in range(min(3, len(self.cache.swa_layers)))
                ]
                logger.info(
                    "[kv-debug][prefill] cache seq_lens (first 3 layers): %s",
                    _kv_lens,
                )

            # Apply repetition penalty to prefill output
            logits = _apply_repetition_penalty(
                logits, generated, repetition_penalty
            )
            last_logits = logits[:, -1]  # [B, vocab]
            next_tok = _sample_next_token(
                last_logits, temperature, top_k, top_p
            )
            if return_log_probs and log_probs_per_batch is not None:
                step_log_probs = paddle.nn.functional.log_softmax(
                    last_logits.cast("float32"), axis=-1
                )  # [B, vocab]
                for b in range(bsz):
                    tok_id = int(next_tok[b, 0].item())
                    log_probs_per_batch[b].append(
                        float(step_log_probs[b, tok_id].item())
                    )
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
                    _tv, _ti = paddle.topk(_d[0], min(5, _d.shape[-1]))
                    logger.info(
                        "[logits-debug][decode step=%d] top-k ids=%s vals=%s",
                        step,
                        _ti.tolist(),
                        [round(v, 3) for v in _tv.tolist()],
                    )
                # Apply repetition penalty
                logits = _apply_repetition_penalty(
                    logits, generated, repetition_penalty
                )
                last_logits = logits[:, -1]  # [B, vocab]
                next_tok = _sample_next_token(
                    last_logits, temperature, top_k, top_p
                )
                if return_log_probs and log_probs_per_batch is not None:
                    step_log_probs = paddle.nn.functional.log_softmax(
                        last_logits.cast("float32"), axis=-1
                    )  # [B, vocab]
                    for b in range(bsz):
                        # Only collect for sequences not yet finished
                        if not bool(done[b, 0].item()):
                            tok_id = int(next_tok[b, 0].item())
                            log_probs_per_batch[b].append(
                                float(step_log_probs[b, tok_id].item())
                            )
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

        if return_log_probs:
            return generated, log_probs_per_batch
        return generated
