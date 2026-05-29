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
N-gram Embedding module for LLM pre-training.

Implements LongCat-style N-gram Embedding Scaling:
- Polynomial rolling hash for N-gram ID computation
- Multiple sub-tables (K splits) per N-gram level to reduce hash collisions
- Linear projection from sub-embedding dim to hidden_size
- Averaged injection: output = (word_emb + sum(proj(ngram_emb))) / normalizer

Reference: Meituan LongCat-Flash-Lite (https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.nn as nn
from paddle import Tensor

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


class NgramEmbedding(nn.Layer):
    """N-gram Embedding with polynomial rolling hash and multi-table lookup.

    Given input_ids [B, S], computes N-gram hash IDs for each (n, k) pair,
    looks up sub-embeddings, projects to hidden_size, and returns the aggregated
    N-gram signal to be added to the standard word embeddings.

    Args:
        config: TransformerConfig with ngram-related fields:
            - ngram_embedding_enabled (bool): master switch
            - ngram_vocab_size_ratio (float): ratio * vocab_size = base ngram vocab M
            - ngram_emb_neighbor_num (int): max N-gram order (N), e.g. 3 means bigram+trigram
            - ngram_emb_split_num (int): number of sub-tables per N-gram level (K)
            - ngram_pad_token_id (int): token ID used for padding shifted sequences
        vocab_size: base vocabulary size of the model
    """

    def __init__(self, config: TransformerConfig, vocab_size: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.vocab_size = vocab_size

        self.m = int(config.ngram_vocab_size_ratio * vocab_size)
        self.k = config.ngram_emb_split_num
        self.n = config.ngram_emb_neighbor_num
        self.pad_token_id = getattr(config, "ngram_pad_token_id", 0)

        num_embedders = self.k * (self.n - 1)
        self.num_embedders = num_embedders

        configured_emb_dim = getattr(config, "ngram_emb_dim", 0)
        if configured_emb_dim > 0:
            self.emb_dim = configured_emb_dim
        else:
            self.emb_dim = self.hidden_size // num_embedders

        self.normalizer = 1 + num_embedders

        embedders = []
        post_projs = []
        for i in range(num_embedders):
            emb_vocab_size = int(self.m + i * 2 + 1)
            emb = nn.Embedding(emb_vocab_size, self.emb_dim)
            proj = nn.Linear(self.emb_dim, self.hidden_size, bias_attr=False)
            embedders.append(emb)
            post_projs.append(proj)

        self.embedders = nn.LayerList(embedders)
        self.post_projs = nn.LayerList(post_projs)

        self._vocab_mods_cache = None

    def _precompute_vocab_mods(self):
        """Precompute (vocab_size^k) mod emb_vocab_dim for each (n, k) pair."""
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache

        vocab_mods = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)

                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * self.vocab_size) % emb_vocab_dim
                    mods.append(power_mod)

                vocab_mods[(i, j)] = mods

        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _compute_shifted_ids(self, input_ids: Tensor) -> dict:
        """Compute right-shifted input_ids for each offset k.

        Args:
            input_ids: [B, S] token IDs

        Returns:
            dict mapping shift amount -> shifted tensor [B, S]
        """
        shifted_ids = {}
        B, S = input_ids.shape

        for k in range(1, self.n):
            shifted = paddle.full([B, k], self.pad_token_id, dtype=input_ids.dtype)
            shifted = paddle.concat([shifted, input_ids[:, :S - k]], axis=1)
            shifted_ids[k + 1] = shifted

        return shifted_ids

    def _compute_ngram_hash_ids(
        self, input_ids: Tensor, shifted_ids: dict, vocab_mods: dict, ngram: int, split_idx: int
    ) -> Tensor:
        """Compute N-gram hash IDs using polynomial rolling hash.

        hash = (input_ids + shifted[2] * (V^1 mod M) + shifted[3] * (V^2 mod M) + ...) mod M

        Args:
            input_ids: [B, S]
            shifted_ids: dict of shift_amount -> [B, S]
            vocab_mods: precomputed (V^k mod M) values
            ngram: N-gram order (2, 3, ...)
            split_idx: sub-table index within this N-gram level

        Returns:
            [B, S] hash IDs
        """
        index = (ngram - 2) * self.k + split_idx
        emb_vocab_dim = int(self.m + index * 2 + 1)
        mods = vocab_mods[(ngram, split_idx)]

        ngram_ids = input_ids.cast("int64")
        for k_offset in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k_offset].cast("int64") * mods[k_offset - 2]

        ngram_ids = ngram_ids % emb_vocab_dim
        return ngram_ids

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute N-gram embedding signal.

        Args:
            input_ids: [B, S] input token IDs

        Returns:
            [B, S, H] N-gram embedding signal (to be added to word embeddings
            and divided by normalizer together)
        """
        vocab_mods = self._precompute_vocab_mods()
        shifted_ids = self._compute_shifted_ids(input_ids)

        ngram_output = paddle.zeros(
            [input_ids.shape[0], input_ids.shape[1], self.hidden_size],
            dtype=self.embedders[0].weight.dtype,
        )

        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j

                hash_ids = self._compute_ngram_hash_ids(
                    input_ids, shifted_ids, vocab_mods, ngram=i, split_idx=j
                )

                x_ngram = self.embedders[index](hash_ids)
                x_proj = self.post_projs[index](x_ngram)
                ngram_output = ngram_output + x_proj

        return ngram_output
