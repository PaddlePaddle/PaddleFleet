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
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor

from paddlefleet import parallel_state, tensor_parallel
from paddlefleet.pipeline_parallel.utils import (
    is_vp_first_stage,
    is_vp_last_stage,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.enums import LayerType
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.paddle_norm import WrappedPaddleNorm
from paddlefleet.transformer.spec_utils import LayerSpec, build_layer
from paddlefleet.transformer.transformer_layer import (
    TransformerLayer,
    get_transformer_layer_offset,
)
from paddlefleet.utils import (
    WrappedTensor,
    get_pg_rank,
)

if TYPE_CHECKING:
    from paddlefleet.packed_seq_params import PackedSeqParams
    from paddlefleet.transformer.transformer_config import TransformerConfig

LayerNormImpl = WrappedPaddleNorm

logger = logging.getLogger(__name__)


def get_num_layers_to_build(
    config: TransformerConfig,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
) -> int:
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        vp_stage (int | None): Virtual pipeline stage number.
        pp_rank (int | None): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    if config.pipeline_model_parallel_layout is not None:
        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    # Fallback for legacy tests.
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    if (
        config.num_layers_in_first_pipeline_stage is not None
        or config.num_layers_in_last_pipeline_stage is not None
    ):
        assert not (
            config.account_for_embedding_in_pipeline_split
            or config.account_for_loss_in_pipeline_split
        ), (
            " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        )
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        # If pp_size <= 2, we do not have any intermediate pipeline stages, and we do not
        # need to check if the left over layers are divisible by the left over stages.
        if pipeline_stages_left > 0:
            assert layers_to_distribute % pipeline_stages_left == 0, (
                "With uneven pipelineing the left over layers must be divisible by left over stages"
            )
            num_layers_per_pipeline_rank = (
                layers_to_distribute // pipeline_stages_left
            )
        else:
            num_layers_per_pipeline_rank = 0

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.

        if (
            is_first_pp_stage
            and config.num_layers_in_first_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = (
                config.num_layers_in_first_pipeline_stage
            )

        if (
            is_last_pp_stage
            and config.num_layers_in_last_pipeline_stage is not None
        ):
            num_layers_per_pipeline_rank = (
                config.num_layers_in_last_pipeline_stage
            )
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert num_layers % config.pipeline_model_parallel_size == 0, (
            "num_layers should be divisible by pipeline_model_parallel_size"
        )
        num_layers_per_pipeline_rank = (
            num_layers // config.pipeline_model_parallel_size
        )

    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        assert num_layers_per_pipeline_rank % vp_size == 0, (
            f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        )
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if config.account_for_embedding_in_pipeline_split:
        if is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, (
                "Not enough layers in the first virtual pipeline stage"
            )

    if config.account_for_loss_in_pipeline_split:
        if is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, (
                "Not enough layers in the last virtual pipeline stage"
            )

    return num_layers_to_build


@dataclass
class TransformerBlockSublayersSpec:
    """
    Dataclass for specifying the sublayers_spec of a transformer block.

    This class defines the structure for configuring the layers and normalization
    within a transformer block, allowing for flexible and customizable architecture designs.

    Args:
        layer_specs (list[LayerSpec] | None): A list of layer specifications for
            the layers within the transformer block. Each specification typically
            defines a complete transformer layer (e.g., self-attention, feed-forward network).
        layer_norm (LayerSpec | paddle.nn.Layer | None): Specification
            or instance of the layer normalization to be applied.
    """

    layer_specs: list[LayerSpec] | None = None
    layer_norm: LayerSpec | paddle.nn.Layer | None = None


def _get_block_sublayers_spec(
    config: TransformerConfig,
    spec: TransformerBlockSublayersSpec | LayerSpec,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
) -> TransformerBlockSublayersSpec:
    """
    Retrieve or construct TransformerBlockSublayersSpec based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (TransformerBlockSublayersSpec | LayerSpec): Specification for the
            transformer block sublayers_spec. Can be either a TransformerBlockSublayersSpec
            instance or a LayerSpec.
        vp_stage (int | None): Virtual pipeline stage number.

    Returns:
        TransformerBlockSublayersSpec: The sublayers_spec for the transformer block.
    """

    # Transformer block sublayers_spec.
    if isinstance(spec, TransformerBlockSublayersSpec):
        return spec

    # LayerSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `TransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, LayerSpec):
        if issubclass(spec.layer, TransformerBlock):
            return spec.sublayers_spec
        elif issubclass(spec.layer, TransformerLayer):
            num_layers = get_num_layers_to_build(config, vp_stage, pp_rank)
            return TransformerBlockSublayersSpec(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
        else:
            raise Exception(f"specialize for {spec.layer.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class TransformerBlock(FleetLayer):
    """Transformer class."""

    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerBlockSublayersSpec | LayerSpec,
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ):
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        pp_group = (
            self.pg_collection.pp if hasattr(self.pg_collection, "pp") else None
        )
        pp_rank = get_pg_rank(pp_group)

        self.sublayers_spec = _get_block_sublayers_spec(
            config, spec, vp_stage, pp_rank
        )
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # required for pipeline parallel schedules
        self.input_tensor = None

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == "selective"
            and "core_attn" in self.config.recompute_layers
        )

        assert self.config.cpu_offloading is False

        self.config._cpu_offloading_context = None

        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)

    def _build_layers(self):
        # Transformer layers.
        def _build_layer(layer_spec, layer_number):
            global_layer_number = layer_number + get_transformer_layer_offset(
                self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
            )  # 1-based index
            if self.config.heterogeneous_block_specs:
                layer_config = self.config.get_config_for_layer(
                    global_layer_number
                )
            else:
                layer_config = self.config

            layer = build_layer(
                layer_spec,
                config=layer_config,
                layer_number=layer_number,
                pg_collection=self.pg_collection,
                vp_stage=self.vp_stage,
            )
            return layer

        # offset is implicit in TransformerLayer
        self.layers = paddle.nn.LayerList(
            [
                _build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.sublayers_spec.layer_specs)
            ]
        )

        # In pipeline parallelism, we want to add this LN only to the last stage of the pipeline
        # self.post_process and self.post_layer_norm guide this behavior
        if (
            self.sublayers_spec.layer_norm
            and self.post_process
            and self.post_layer_norm
        ):
            self.final_layernorm = build_layer(
                self.sublayers_spec.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None  # Either this or nn.Identity

    def _get_layer(self, layer_number: int):
        return self.layers[layer_number]

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            ):
                for index in range(start, end):
                    layer = self._get_layer(index)

                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            """Determines whether to use the `tensor_parallel.checkpoint`"""
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            )

        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(
                        layer_idx, layer_idx + self.config.recompute_num_layers
                    )
                )

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                hidden_states, context = custom(layer_idx, layer_idx + 1)(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self,
        hidden_states: Tensor | WrappedTensor,
        attention_mask: Tensor,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: Tensor | None = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Tensor | WrappedTensor): Input tensor of shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor | None): Context tensor for cross-attention.
            context_mask (Tensor | None): Mask for cross-attention context
            rotary_pos_emb (Tensor | None): Rotary positional embeddings.
            rotary_pos_cos (Tensor | None): Rotary embedding cosine.
            rotary_pos_sin (Tensor | None): Rotary embedding sine.
            attention_bias (Tensor | None): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
            packed_seq_params (PackedSeqParams | None): Parameters for packed sequence
                processing.

        Returns:
            Tensor | Tuple[Tensor, Tensor]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            hidden_states = self.input_tensor

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # Forward pass.
            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    hidden_states, context = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        rotary_pos_cos=rotary_pos_cos,
                        rotary_pos_sin=rotary_pos_sin,
                        attention_bias=attention_bias,
                        packed_seq_params=packed_seq_params,
                    )

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        # If this TransformerBlock is empty, input and output hidden states will be the same node
        # on the computational graph and will lead to unexpected errors in pipeline schedules.
        if (
            not self.pre_process
            and len(self.layers) == 0
            and not self.final_layernorm
        ):
            hidden_states = hidden_states.clone()

        return hidden_states
