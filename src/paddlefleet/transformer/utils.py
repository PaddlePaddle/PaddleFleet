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

"""Utilities for transformer layers."""

from __future__ import annotations

from functools import lru_cache

import paddle


@lru_cache(maxsize=32)
def get_default_causal_mask(sq: int) -> paddle.Tensor:
    """Return the causal upper triangular mask for softmax input."""
    return paddle.triu(paddle.ones(sq, sq), diagonal=1).bool()


@lru_cache(maxsize=32)
def get_sliding_window_causal_mask(sq, skv, sliding_window):
    """Create the equivalent attention mask for SWA in [sq, skv] shape"""
    m = paddle.ones(sq, skv, dtype=paddle.bool)
    mu = paddle.triu(m, diagonal=skv - sq - sliding_window[0])
    ml = paddle.tril(mu, diagonal=skv - sq + sliding_window[1])
    ml = ~ml

    return ml


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def is_layer_window_attention(
    sliding_window: tuple[int, int] | None,
    window_attn_skip_freq: int | list,
    layer_number: int,
) -> bool:
    # layer_number is 1-indexed
    if not sliding_window:
        return False
    if window_attn_skip_freq is None:
        return True
    if isinstance(window_attn_skip_freq, int):
        return layer_number % window_attn_skip_freq != 0
    if isinstance(window_attn_skip_freq, list):
        return bool(window_attn_skip_freq[layer_number - 1])

    raise ValueError(
        f"Invalid `window_attn_skip_freq`: {type(window_attn_skip_freq)}, "
        f"{window_attn_skip_freq}"
    )
