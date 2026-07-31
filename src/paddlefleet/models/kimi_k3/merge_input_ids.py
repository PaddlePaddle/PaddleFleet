# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Kimi-K3 image feature / text token merging."""

from __future__ import annotations

import paddle


def _where_2d(cond: paddle.Tensor):
    """Return (row_indices, col_indices) for a 2-D boolean tensor"""
    nz = paddle.nonzero(cond)  # [M, 2]
    if nz.shape[0] == 0:
        empty = paddle.zeros([0], dtype="int64")
        return empty, empty
    return nz[:, 0], nz[:, 1]


def merge_input_ids_with_image_features(
    image_features,
    inputs_embeds: paddle.Tensor,
    input_ids: paddle.Tensor,
    attention_mask: paddle.Tensor,
    image_token_index: int,
    pad_token_id: int,
    ignore_index: int = -100,
    labels: paddle.Tensor | None = None,
):
    """Merge dynamically-expanded image features into the text embedding
    sequence.

    Args:
        image_features: list of per-image tensors ``(num_tokens_i, embed_dim)``
            in the same order the image placeholders appear (row-major over the
            batch), or a single ``(num_tokens, embed_dim)`` tensor for one image.
        inputs_embeds: ``(batch, seq, embed_dim)`` text embeddings.
        input_ids: ``(batch, seq)``.
        attention_mask: ``(batch, seq)``.
        image_token_index: the media placeholder token id.
        pad_token_id: pad token id.
        ignore_index: label ignore index.
        labels: optional ``(batch, seq)``.

    Returns:
        (final_embedding, final_attention_mask, final_labels, position_ids)
        where sequences are expanded to ``max_embed_dim``.
    """
    if isinstance(image_features, (list, tuple)):
        feature_lengths = [int(x.shape[0]) for x in image_features]
        image_features = paddle.concat(image_features, axis=0)
    else:
        feature_lengths = [int(image_features.shape[0])]

    embed_dim = image_features.shape[-1]
    batch_size, sequence_length = input_ids.shape

    # left_padding is True when no sample ends with a pad token.
    left_padding = not bool(
        paddle.sum((input_ids[:, -1] == pad_token_id).astype("int64")).item()
    )

    # 1. Per-token occupation: image placeholders occupy feature_lengths tokens.
    flat_ids = input_ids.flatten()
    occupation = paddle.ones_like(flat_ids)
    img_positions = paddle.nonzero(flat_ids == image_token_index).flatten()
    if img_positions.shape[0] > 0:
        updates = paddle.to_tensor(feature_lengths, dtype=occupation.dtype)
        occupation = paddle.scatter(occupation, img_positions, updates)
    occupation = occupation.reshape(input_ids.shape)

    max_embed_dim = int(occupation.sum(-1).max().item())
    assert max_embed_dim >= sequence_length, (
        f"max_embed_dim ({max_embed_dim}) < sequence_length ({sequence_length})"
    )

    batch_indices, non_image_indices = _where_2d(input_ids != image_token_index)

    # 2. Target positions for text tokens in the expanded sequence.
    new_token_positions = paddle.cumsum(occupation, axis=-1) - 1
    nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
    if left_padding:
        new_token_positions = new_token_positions + nb_image_pad.unsqueeze(1)

    text_to_overwrite = paddle.gather_nd(
        new_token_positions,
        paddle.stack([batch_indices, non_image_indices], axis=-1),
    )

    # 3. Allocate expanded buffers.
    final_embedding = paddle.zeros(
        [batch_size, max_embed_dim, embed_dim], dtype=inputs_embeds.dtype
    )
    final_attention_mask = paddle.zeros(
        [batch_size, max_embed_dim], dtype=attention_mask.dtype
    )
    final_labels = None
    if labels is not None:
        final_labels = paddle.full(
            [batch_size, max_embed_dim], ignore_index, dtype=input_ids.dtype
        )

    # 4. Scatter text embeddings / mask / labels into their new positions.
    if batch_indices.shape[0] > 0:
        text_src = paddle.gather_nd(
            inputs_embeds,
            paddle.stack([batch_indices, non_image_indices], axis=-1),
        )
        final_embedding = paddle.index_put(
            final_embedding, (batch_indices, text_to_overwrite), text_src
        )
        mask_src = paddle.gather_nd(
            attention_mask,
            paddle.stack([batch_indices, non_image_indices], axis=-1),
        )
        final_attention_mask = paddle.index_put(
            final_attention_mask,
            (batch_indices, text_to_overwrite),
            mask_src,
        )
        if labels is not None:
            label_src = paddle.gather_nd(
                labels,
                paddle.stack([batch_indices, non_image_indices], axis=-1),
            )
            final_labels = paddle.index_put(
                final_labels, (batch_indices, text_to_overwrite), label_src
            )

    # 5. Everything not written by text is an image slot.
    image_to_overwrite = paddle.full(
        [batch_size, max_embed_dim], True, dtype="bool"
    )
    if batch_indices.shape[0] > 0:
        image_to_overwrite = paddle.index_put(
            image_to_overwrite,
            (batch_indices, text_to_overwrite),
            paddle.zeros([batch_indices.shape[0]], dtype="bool"),
        )
    # Each row has ``nb_image_pad`` unwritten slots that belong to padding
    # rather than to an image, and they sit on the padding side.
    slot_rank = paddle.cumsum(image_to_overwrite.astype("int64"), axis=-1) - 1
    if left_padding:
        pad_bound = slot_rank >= nb_image_pad.unsqueeze(1)
    else:
        n_slots = image_to_overwrite.astype("int64").sum(-1, keepdim=True)
        pad_bound = slot_rank < (n_slots - nb_image_pad.unsqueeze(1))
    image_to_overwrite = paddle.logical_and(image_to_overwrite, pad_bound)

    n_image_slots = int(image_to_overwrite.astype("int64").sum().item())
    if n_image_slots != image_features.shape[0]:
        raise ValueError(
            f"Mismatch: {n_image_slots} image slots vs "
            f"{image_features.shape[0]} image feature tokens."
        )

    mask3 = image_to_overwrite.unsqueeze(-1).expand(
        [batch_size, max_embed_dim, embed_dim]
    )
    final_embedding = final_embedding.masked_scatter(
        mask3,
        image_features.reshape([-1, embed_dim]).astype(final_embedding.dtype),
    )

    final_attention_mask = paddle.maximum(
        final_attention_mask,
        image_to_overwrite.astype(final_attention_mask.dtype),
    )
    position_ids = (
        paddle.cumsum(final_attention_mask.astype("int64"), axis=-1) - 1
    )
    position_ids = paddle.where(
        final_attention_mask == 0,
        paddle.ones_like(position_ids),
        position_ids,
    )

    # 6. Zero out embeddings at original pad positions.
    pad_batch, pad_cols = _where_2d(input_ids == pad_token_id)
    if pad_batch.shape[0] > 0:
        indices_to_mask = paddle.gather_nd(
            new_token_positions,
            paddle.stack([pad_batch, pad_cols], axis=-1),
        )
        zero_src = paddle.zeros(
            [pad_batch.shape[0], embed_dim], dtype=final_embedding.dtype
        )
        final_embedding = paddle.index_put(
            final_embedding, (pad_batch, indices_to_mask), zero_src
        )

    return final_embedding, final_attention_mask, final_labels, position_ids
