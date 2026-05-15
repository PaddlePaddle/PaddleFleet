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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

    from paddlefleet.transformer.transformer_config import TransformerConfig

import paddle
from paddle import Tensor

from paddlefleet import tensor_parallel
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.utils import get_tensor_model_parallel_group_if_none

logger = logging.getLogger(__name__)


class LanguageModelEmbedding(FleetLayer):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head. Defaults to 0.
        scatter_to_sequence_parallel (bool): Set to False to disable scatter of embedding
            across sequence parallel region. Defaults to True.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "none"
        ] = "learned_absolute",
        num_tokentypes: int = 0,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Group | None = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = (
            position_embedding_type == "learned_absolute"
        )
        self.sequence_parallel = self.config.sequence_parallel
        self.num_tokentypes = num_tokentypes
        self.scatter_to_sequence_parallel = scatter_to_sequence_parallel
        if self.sequence_parallel:
            assert self.scatter_to_sequence_parallel is True, (
                "If sequence parallel is turned on, scatter_to_sequence_parallel "
                "must be set to True."
            )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group)
        self.reduce_scatter_embeddings = (
            (not self.add_position_embedding)
            and self.num_tokentypes <= 0
            and self.sequence_parallel
            and self.scatter_to_sequence_parallel
        )

        # Word embeddings (parallel).
        self.embed_tokens = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.embedding_init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
            tp_group=self.tp_group,
        )

        # Position embedding (serial).
        if self.add_position_embedding:
            self.position_embeddings = paddle.nn.Embedding(
                self.max_sequence_length, self.config.hidden_size
            )

            # Initialize the position embeddings.
            if self.config.perform_initialization:
                self.config.embedding_init_method(
                    self.position_embeddings.weight
                )

        if self.num_tokentypes > 0:
            self.tokentype_embeddings = paddle.nn.Embedding(
                self.num_tokentypes, self.config.hidden_size
            )
            # Initialize the token-type embeddings.
            if self.config.perform_initialization:
                self.config.embedding_init_method(
                    self.tokentype_embeddings.weight
                )
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = paddle.nn.Dropout(
            self.config.hidden_dropout_prob
        )

        # ============ Per-Layer Embeddings (PLE) ============
        # Matches Gemma 4 architecture:
        # - embed_tokens_per_layer: scaled by sqrt(per_layer_dim) after lookup
        # - per_layer_model_projection: Linear(H, L*D), scaled by 1/sqrt(H) after projection
        # - per_layer_projection_norm: RMSNorm(per_layer_dim) on each layer slice
        # - combine: (normed_projection + scaled_lookup) * 1/sqrt(2)
        self.use_per_layer_embeddings = getattr(
            self.config, 'use_per_layer_embeddings', False
        )
        if self.use_per_layer_embeddings:
            num_layers = self.config.num_hidden_layers
            per_layer_dim = getattr(self.config, 'per_layer_dim', 256)

            # Per-layer embedding table: [vocab_size, num_layers * per_layer_dim]
            # Output is scaled by sqrt(per_layer_dim) to match Gemma4 ScaledWordEmbedding
            self.embed_tokens_per_layer = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=vocab_size,
                embedding_dim=num_layers * per_layer_dim,
                init_method=self.config.embedding_init_method,
                reduce_scatter_embeddings=False,  # No scatter for PLE
                config=self.config,
                tp_group=self.tp_group,
            )
            # Gemma4: embed_scale = per_layer_dim ** 0.5
            self._ple_embed_scale = per_layer_dim ** 0.5

            # Project main embedding to per-layer space: H -> L*D
            self.per_layer_model_projection = paddle.nn.Linear(
                self.config.hidden_size,
                num_layers * per_layer_dim,
                bias_attr=False,
            )
            # Gemma4: per_layer_model_projection_scale = hidden_size ** -0.5
            self._ple_proj_scale = self.config.hidden_size ** -0.5

            # RMSNorm on per_layer_dim (applied per-layer after reshape)
            # Gemma4: per_layer_projection_norm = RMSNorm(per_layer_dim)
            self._ple_norm_weight = paddle.create_parameter(
                shape=[per_layer_dim],
                dtype=paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.Constant(1.0),
            )
            self._ple_norm_eps = self.config.rms_norm_eps

            # Gemma4: per_layer_input_scale = 2.0 ** -0.5
            self._ple_input_scale = 2.0 ** -0.5

            self._per_layer_dim = per_layer_dim
            self._num_layers = num_layers

            logger.info(
                f"[Gemma 4 PLE] Enabled: per_layer_dim={per_layer_dim}, "
                f"num_layers={num_layers}, vocab_size={vocab_size}"
            )

    @property
    def embedding_weight(self):
        return self.embed_tokens.weight

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.embed_tokens.weight.data.fill_(0)
        self.embed_tokens.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        tokentype_ids: int | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass of the embedding layer.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is
                set to True. Defaults to None

        Returns:
            Tensor: The output embeddings, or tuple of (embeddings, per_layer_inputs)
        """
        embed_tokens = self.embed_tokens(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embed_tokens + position_embeddings
        else:
            embeddings = embed_tokens

        # ============ Per-Layer Embeddings (PLE) ============
        # Matches Gemma 4 pipeline:
        # 1. get_per_layer_inputs: embed_tokens_per_layer(ids) * sqrt(D), reshape [B,S,L,D]
        # 2. project_per_layer_inputs: proj(embeds) * 1/sqrt(H), reshape [B,S,L,D], RMSNorm(D)
        # 3. combine: (normed_projection + token_embedding) * 1/sqrt(2)
        #
        # NOTE: Wrapped in recompute to avoid inplace version errors.
        # per_layer_model_projection creates a MatmulGradNode; the sharding
        # optimizer's parameter sync bumps weight versions between steps.
        # Without recompute the stale version snapshot triggers
        # "Tensor modified by inplace operation" during backward.
        per_layer_inputs = None
        if self.use_per_layer_embeddings:
            from paddle.distributed.fleet.recompute import recompute

            per_layer_inputs = recompute(
                self._compute_per_layer_inputs,
                input_ids,
                embeddings,
            )

        if (
            not self.reduce_scatter_embeddings
            and self.sequence_parallel
            and self.scatter_to_sequence_parallel
        ):
            # Data format change to avoid explicit transposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose([1, 0, 2]).contiguous()
            # Also transpose per_layer_inputs if needed
            if per_layer_inputs is not None:
                per_layer_inputs = per_layer_inputs.transpose([1, 0, 2, 3]).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            # tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids)
            if self.sequence_parallel and self.scatter_to_sequence_parallel:
                tokentype_embedding = tokentype_embedding.permute(
                    1, 0, 2
                ).contiguous()
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.sequence_parallel:
            if (
                not self.reduce_scatter_embeddings
                and self.scatter_to_sequence_parallel
            ):
                embeddings = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        embeddings, group=self.tp_group
                    )
                )
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if (
                self.config.clone_scatter_output_in_embedding
                and self.scatter_to_sequence_parallel
            ):
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        if per_layer_inputs is not None:
            return embeddings, per_layer_inputs
        return embeddings

    def _compute_per_layer_inputs(self, input_ids, embeddings):
        """Compute PLE per-layer inputs (called inside recompute boundary)."""
        import paddle.nn.functional as F_emb

        batch_size, seq_len = input_ids.shape

        # 1. Token-identity component: lookup + scale by sqrt(per_layer_dim)
        ple_lookup = self.embed_tokens_per_layer(input_ids) * self._ple_embed_scale
        # Reshape to [B, S, L, D]
        ple_lookup = ple_lookup.reshape(
            [batch_size, seq_len, self._num_layers, self._per_layer_dim]
        )

        # 2. Context-aware component: project + scale by 1/sqrt(hidden_size)
        ple_projection = self.per_layer_model_projection(embeddings) * self._ple_proj_scale
        # Reshape to [B, S, L, D]
        ple_projection = ple_projection.reshape(
            [batch_size, seq_len, self._num_layers, self._per_layer_dim]
        )
        # RMSNorm on last dim (per_layer_dim), applied to projection only
        _norm_w = self._ple_norm_weight
        if ple_projection.dtype != _norm_w.dtype:
            _norm_w = _norm_w.cast(ple_projection.dtype)
        ple_projection = F_emb.rms_norm(
            ple_projection, ple_projection.shape[-1:],
            _norm_w, self._ple_norm_eps
        )

        # 3. Combine: (projection + token_emb) * 1/sqrt(2)
        return (ple_projection + ple_lookup) * self._ple_input_scale
