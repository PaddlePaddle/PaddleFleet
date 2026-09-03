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
Byte-level Embedding module for LLM pre-training.

For each BPE token, looks up embeddings for its constituent bytes and
aggregates via mean pooling. The token-to-bytes mapping is prebuilt from
the tokenizer at initialization time and stored as a non-trainable buffer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import paddle
import paddle.nn as nn
from paddle import Tensor

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)

# All possible byte values (0-255) + 1 padding index
BYTE_VOCAB_SIZE = 256
BYTE_PAD_IDX = 256  # index 256 is used for padding


class ByteEmbedding(nn.Layer):
    """Byte-level embedding with per-token mean pooling.

    At model initialization, builds a token_id -> bytes mapping table from the
    tokenizer (registered as a buffer). During forward, for each token, looks up
    all its byte embeddings and mean-pools them into a single vector.

    Args:
        config: TransformerConfig with byte_embedding_tokenizer_path field.
        vocab_size: base vocabulary size of the model.
    """

    def __init__(self, config: TransformerConfig, vocab_size: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.vocab_size = vocab_size

        # Byte embedding table: 256 byte values + 1 padding index
        self.byte_emb = nn.Embedding(
            BYTE_VOCAB_SIZE + 1, self.hidden_size, padding_idx=BYTE_PAD_IDX
        )

        # Build the token -> bytes mapping from tokenizer
        tokenizer_path = getattr(config, "byte_embedding_tokenizer_path", "")
        token_to_bytes, token_byte_lengths = self._build_byte_mapping(
            tokenizer_path, vocab_size
        )

        # Register as buffers (non-trainable, moves with device)
        self.register_buffer("token_to_bytes", token_to_bytes)
        self.register_buffer("token_byte_lengths", token_byte_lengths)

    def _build_byte_mapping(
        self, tokenizer_path: str, vocab_size: int
    ) -> tuple[Tensor, Tensor]:
        """Build token_id -> bytes mapping from tokenizer.

        Args:
            tokenizer_path: Path to the tokenizer directory/file.
            vocab_size: Number of tokens in the vocabulary.

        Returns:
            token_to_bytes: [vocab_size, max_bytes] padded byte indices.
            token_byte_lengths: [vocab_size] number of valid bytes per token.
        """
        try:
            from ernie_core.tokenizers.tokenization_eb_utf16be import (
                ErnieBotTokenizer,
            )
            tokenizer = ErnieBotTokenizer.from_pretrained(tokenizer_path)
        except ImportError:
            try:
                from paddlenlp.transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except ImportError:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        logger.info(
            f"ByteEmbedding: loading tokenizer from '{tokenizer_path}' "
            f"to build byte mapping for vocab_size={vocab_size}"
        )

        # Build special token ID set for filtering
        special_ids = self._collect_special_ids(tokenizer)
        if special_ids:
            logger.info(
                f"ByteEmbedding: filtering {len(special_ids)} special tokens"
            )

        # Collect bytes for each token
        all_bytes = []
        max_byte_len = 0

        for token_id in range(vocab_size):
            # Special tokens get zero-signal (padding fallback)
            if token_id in special_ids:
                byte_list = [0]
            else:
                try:
                    token_str = tokenizer._convert_id_to_token(token_id)
                    if token_str is None:
                        token_str = ""
                    # Heuristic: skip tokens that look like special markers
                    if self._is_special_token_str(token_str):
                        byte_list = [0]
                    else:
                        # Convert internal token representation to actual text
                        # (handles SentencePiece '▁', GPT 'Ġ', byte fallbacks, etc.)
                        token_text = self._token_to_text(
                            tokenizer, token_id, token_str
                        )
                        byte_list = list(token_text.encode("utf-8"))
                except Exception:
                    byte_list = []

            # Ensure at least 1 byte (fallback to avoid division by zero)
            if len(byte_list) == 0:
                byte_list = [0]

            all_bytes.append(byte_list)
            max_byte_len = max(max_byte_len, len(byte_list))

        # Build padded tensors
        token_to_bytes = paddle.full(
            [vocab_size, max_byte_len], BYTE_PAD_IDX, dtype="int32"
        )
        token_byte_lengths = paddle.zeros([vocab_size], dtype="int32")

        for token_id, byte_list in enumerate(all_bytes):
            length = len(byte_list)
            token_byte_lengths[token_id] = length
            for j, b in enumerate(byte_list):
                token_to_bytes[token_id, j] = b

        logger.info(
            f"ByteEmbedding: mapping built. max_bytes_per_token={max_byte_len}"
        )

        return token_to_bytes, token_byte_lengths

    @staticmethod
    def _token_to_text(tokenizer, token_id: int, token_str: str) -> str:
        """Convert internal token representation to actual text.

        Handles SentencePiece '▁' prefix, GPT 'Ġ' markers, byte-level
        fallback tokens, etc. by using the tokenizer's decoding pipeline
        rather than the raw internal piece string.
        """
        # Prefer convert_tokens_to_string (handles SP ▁ → space, etc.)
        if hasattr(tokenizer, "convert_tokens_to_string"):
            text = tokenizer.convert_tokens_to_string([token_str])
            if text is not None:
                return text
        # Fallback: decode the token_id directly
        if hasattr(tokenizer, "decode"):
            try:
                text = tokenizer.decode([token_id], skip_special_tokens=False)
                if text is not None:
                    return text
            except Exception:
                pass
        # Last resort: use the raw token string
        return token_str

    @staticmethod
    def _collect_special_ids(tokenizer) -> set[int]:
        """Collect special token IDs from tokenizer using multiple strategies."""
        special_ids: set[int] = set()

        # Strategy 1: tokenizer.all_special_ids (HuggingFace standard)
        try:
            ids = tokenizer.all_special_ids
            if ids:
                special_ids.update(ids)
        except (AttributeError, TypeError):
            pass

        # Strategy 2: added_tokens_encoder (dynamically added tokens)
        try:
            added = tokenizer.added_tokens_encoder
            if added:
                special_ids.update(added.values())
        except (AttributeError, TypeError):
            pass

        return special_ids

    @staticmethod
    def _is_special_token_str(token_str: str) -> bool:
        """Heuristic check for special token patterns like <|xxx|>, <xxx>, [xxx]."""
        if not token_str:
            return True
        # Pattern: <|...|> (e.g., <|endoftext|>, <|image|>)
        if token_str.startswith("<|") and token_str.endswith("|>"):
            return True
        # Pattern: <...> with no spaces (e.g., <s>, </s>, <unk>, <pad>)
        if (
            token_str.startswith("<")
            and token_str.endswith(">")
            and " " not in token_str
            and len(token_str) <= 20
        ):
            return True
        # Pattern: [xxx] (e.g., [CLS], [SEP], [PAD])
        if (
            token_str.startswith("[")
            and token_str.endswith("]")
            and " " not in token_str
            and len(token_str) <= 20
        ):
            return True
        return False

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute byte-level embedding signal via mean pooling.

        Args:
            input_ids: [B, S] input token IDs.

        Returns:
            [B, S, hidden_size] byte embedding signal.
        """
        # input_ids: [B, S]
        B, S = input_ids.shape

        # Look up byte indices for each token: [B, S, max_bytes]
        # token_to_bytes is [vocab_size, max_bytes]
        flat_ids = input_ids.reshape([-1])  # [B*S]
        byte_indices = paddle.index_select(
            self.token_to_bytes, flat_ids, axis=0
        )  # [B*S, max_bytes]
        byte_indices = byte_indices.reshape([B, S, -1])  # [B, S, max_bytes]

        # Look up byte embeddings: [B, S, max_bytes, hidden_size]
        byte_embeds = self.byte_emb(byte_indices)

        # Build mask from token_byte_lengths: [B, S, max_bytes]
        lengths = paddle.index_select(
            self.token_byte_lengths, flat_ids, axis=0
        )  # [B*S]
        lengths = lengths.reshape([B, S])  # [B, S]

        max_bytes = byte_indices.shape[-1]
        # Create range tensor [1, 1, max_bytes]
        range_tensor = paddle.arange(max_bytes, dtype="int32").reshape(
            [1, 1, max_bytes]
        )
        # mask: [B, S, max_bytes], True where byte position is valid
        mask = range_tensor < lengths.unsqueeze(-1)  # [B, S, max_bytes]

        # Apply mask and mean pool
        mask_float = mask.astype(byte_embeds.dtype).unsqueeze(-1)  # [B, S, max_bytes, 1]
        masked_embeds = byte_embeds * mask_float  # [B, S, max_bytes, hidden_size]

        # Sum over max_bytes dim and divide by lengths
        summed = masked_embeds.sum(axis=2)  # [B, S, hidden_size]
        lengths_float = lengths.astype(summed.dtype).unsqueeze(-1)  # [B, S, 1]
        output = summed / lengths_float  # [B, S, hidden_size]

        return output
