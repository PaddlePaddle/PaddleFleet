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

"""Incremental decode cache for CSA / HCA (DSv4 Hybrid Attention).

Standard attention keeps a single growing ``(k, v)`` per layer. CSA/HCA needs
more: besides the raw single-head MQA KV (for the sliding window), it maintains
a *compressed* KV stream produced by a gated-pooling compressor over groups of
``compress_ratio`` raw tokens. Decoding one token at a time therefore requires:

* the raw single-head KV (append 1 per step; drives the sliding window),
* the compressed KV (append 1 whenever a group of ``ratio`` tokens closes),
* a buffer of the hidden states for the *current, not-yet-closed* group (and,
  for the overlapping ratio==4 path, the previous closed group) so the next
  compressed token can be emitted with the exact same math as prefill,
* (ratio==4 only) the indexer's own compressed keys.

:class:`CSADynamicCache` holds one :class:`_CSALayerState` per ``layer_idx`` and
is duck-typed by ``CompressedSparseAttention.forward``; no import of this module
is required from the attention code. It also exposes the standard
``update(k, v, layer_idx)`` / ``get_seq_len`` protocol so a model that mixes
CSA layers with standard-attention layers can share a single cache object.
"""

from __future__ import annotations

import paddle


class _CSALayerState:
    """Per-layer incremental decode state for one CSA/HCA attention layer."""

    def __init__(self) -> None:
        # Raw single-head MQA KV: [b, kv_len, v_head_dim].
        self.raw_kv: paddle.Tensor | None = None
        # Compressed KV stream: [b, n_compressed, v_head_dim].
        self.compressed_kv: paddle.Tensor | None = None
        # Indexer compressed keys (ratio==4 only): [b, n_compressed, index_hd].
        self.idx_compressed_k: paddle.Tensor | None = None
        # Hidden states of the current (open) group: [b, cur_len, hidden].
        self.x_cur: paddle.Tensor | None = None
        # Hidden states of the previous closed group (overlap): [b, ratio, hidden].
        self.x_prev: paddle.Tensor | None = None
        # Standard-attention fallback tensors (for mixed models).
        self.k: paddle.Tensor | None = None
        self.v: paddle.Tensor | None = None

    # -- raw KV / position bookkeeping --------------------------------------

    def raw_seq_len(self) -> int:
        """Number of raw tokens already cached (== next token's position)."""
        if self.raw_kv is None:
            return 0
        return self.raw_kv.shape[1]

    @property
    def n_compressed(self) -> int:
        if self.compressed_kv is None:
            return 0
        return self.compressed_kv.shape[1]

    def append_raw(self, k_tok: paddle.Tensor) -> None:
        """Append one raw single-head KV token: k_tok is [b, 1, v_head_dim]."""
        if self.raw_kv is None:
            self.raw_kv = k_tok
        else:
            self.raw_kv = paddle.concat([self.raw_kv, k_tok], axis=1)

    def append_x(self, x_tok: paddle.Tensor) -> None:
        """Accumulate one hidden-state token into the open group buffer."""
        if self.x_cur is None:
            self.x_cur = x_tok
        else:
            self.x_cur = paddle.concat([self.x_cur, x_tok], axis=1)

    def append_compressed(self, comp_tok: paddle.Tensor) -> None:
        """Append one compressed KV token: comp_tok is [b, 1, v_head_dim]."""
        if self.compressed_kv is None:
            self.compressed_kv = comp_tok
        else:
            self.compressed_kv = paddle.concat(
                [self.compressed_kv, comp_tok], axis=1
            )

    def append_idx_compressed(self, idx_tok: paddle.Tensor) -> None:
        """Append one indexer compressed key: idx_tok is [b, 1, index_hd]."""
        if self.idx_compressed_k is None:
            self.idx_compressed_k = idx_tok
        else:
            self.idx_compressed_k = paddle.concat(
                [self.idx_compressed_k, idx_tok], axis=1
            )

    def roll_group(self) -> None:
        """Close the current group: it becomes ``x_prev``; reset ``x_cur``."""
        self.x_prev = self.x_cur
        self.x_cur = None

    # -- standard-attention protocol (mixed models) -------------------------

    def update_std(
        self, k_new: paddle.Tensor, v_new: paddle.Tensor
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = paddle.concat([self.k, k_new], axis=1)
            self.v = paddle.concat([self.v, v_new], axis=1)
        return self.k, self.v


class CSADynamicCache:
    """Per-layer decode cache dispatching by ``layer_idx``.

    Supports both the CSA/HCA incremental protocol (via :meth:`get_csa_state`)
    and the standard ``update(k, v, layer_idx)`` protocol so a single cache can
    serve a model that interleaves CSA and standard-attention layers.
    """

    def __init__(self, num_layers: int) -> None:
        self._states: list[_CSALayerState] = [
            _CSALayerState() for _ in range(num_layers)
        ]
        # CSA/HCA layers do their own windowing inside the attention op, so the
        # generator's SWA KV-cache truncation never applies here. Expose the
        # attributes anyway to match ``DynamicKVCache``'s interface.
        self.swa_layers: list[bool] = [False] * num_layers
        self.window_size: int | None = None

    def get_csa_state(self, layer_idx: int) -> _CSALayerState:
        return self._states[layer_idx]

    def has_layer_cache(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` already cached something (prefill has run).

        Mirrors :meth:`DynamicKVCache.has_layer_cache` so that
        ``_is_incremental_decode`` in CSA/MLA attention correctly identifies
        decode steps vs the first prefill pass.

        A CSA layer is considered "primed" once ``raw_kv`` is populated
        (set by ``_prime_cache_prefill``). A standard-attention layer in the
        same model is primed once ``k`` is populated (set by ``update``).
        """
        st = self._states[layer_idx]
        return st.raw_kv is not None or st.k is not None

    # -- standard-attention protocol ----------------------------------------

    def get_seq_len(self, layer_idx: int = 0) -> int:
        st = self._states[layer_idx]
        if st.k is not None:
            return st.k.shape[1]
        if st.raw_kv is not None:
            return st.raw_kv.shape[1]
        for other in self._states:
            if other.k is not None:
                return other.k.shape[1]
            if other.raw_kv is not None:
                return other.raw_kv.shape[1]
        return 0

    def update(
        self, k_new: paddle.Tensor, v_new: paddle.Tensor, layer_idx: int
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        return self._states[layer_idx].update_std(k_new, v_new)

    def reset(self) -> None:
        for i in range(len(self._states)):
            self._states[i] = _CSALayerState()
