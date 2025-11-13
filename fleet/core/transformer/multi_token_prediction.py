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

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import paddle
from paddle import Tensor

from fleet.core import parallel_state, tensor_parallel
from fleet.core.models.backends import BackendSpecProvider, LocalSpecProvider
from fleet.core.pipeline_parallel.utils import is_vp_last_stage
from fleet.core.process_groups_config import ProcessGroupCollection
from fleet.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from fleet.core.transformer.enums import AttnMaskType
from fleet.core.transformer.layer import FleetLayer
from fleet.core.transformer.spec_utils import LayerSpec, build_layer

if TYPE_CHECKING:
    from fleet.core.packed_seq_params import PackedSeqParams
    from fleet.core.transformer.transformer_block import (
        TransformerBlockSublayers,
    )
    from fleet.core.transformer.transformer_config import TransformerConfig

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]


def roll_tensor(tensor, shifts=-1, dims=-1, cp_group=None):
    """Roll the tensor input along the sequence dimension with Context Parallelism (CP) support.

    This function extends the original roll_tensor to support Context Parallelism, which allows
    MTP to work with CP > 1. When CP is enabled, the sequence dimension is split across CP ranks,
    and tensor rolling requires communication between adjacent CP ranks to properly handle the
    boundary conditions.

    For CP=1 (default behavior): Uses standard paddle.roll with zero padding
    For CP>1: Splits tensor into chunks, performs rolling within each chunk, then exchanges
    boundary elements between adjacent CP ranks to maintain sequence continuity.

    Args:
        tensor (Tensor): The input tensor to roll.
        shifts (int): The shift of the tensor (typically -1 for MTP).
        dims (int): The dimension to roll (typically -1 for sequence dimension).
        cp_group (ProcessGroup): The context parallelism process group. If None or size=1,
                               falls back to standard rolling behavior.
    Returns:
        tuple: (rolled_tensor, sum_of_rolled_tensor)
    """
    # Standard rolling behavior when CP is not enabled (cp_group is None or size=1)
    if cp_group is None or cp_group.size() == 1:
        rolled_tensor = paddle.roll(tensor, shifts=shifts, dims=dims)
        rolled_tensor.select(dims, shifts).fill_(0)
        return rolled_tensor, rolled_tensor.sum()

    # CP-enabled rolling: Split tensor into chunks and handle boundary communication
    # This matches the batch splitting logic in get_batch_on_this_cp_rank() function
    tensor_list = tensor.chunk(2, dim=dims)
    rolled_tensor_list = []
    for i in range(len(tensor_list)):
        rolled_tensor_list.append(
            paddle.roll(tensor_list[i], shifts=shifts, dims=dims)
        )

    # Prepare tensors for communication between CP ranks
    # Each CP rank needs to send boundary elements to adjacent ranks
    tensor_send_list = []
    tensor_recv_list = []
    for i in range(len(rolled_tensor_list)):
        tensor_send_list.append(
            rolled_tensor_list[i].select(dims, shifts).contiguous()
        )
        empty_tensor = paddle.empty(
            tensor_send_list[i].shape,
            dtype=tensor_send_list[i].dtype,
            device=paddle.cuda.current_device(),
        )
        tensor_recv_list.append(empty_tensor)

    # Get the global rank of next and prev process in the cp group
    global_ranks = paddle.distributed.get_process_group_ranks(group=cp_group)
    local_rank = paddle.distributed.get_rank(group=cp_group)
    next_rank = global_ranks[(local_rank + 1) % len(global_ranks)]
    prev_rank = global_ranks[(local_rank - 1) % len(global_ranks)]

    # Start send and recv ops
    ops = []
    if local_rank != 0:
        req_send_first_part = paddle.distributed.isend(
            tensor=tensor_send_list[0], dst=prev_rank
        )
        ops.append(req_send_first_part)
        req_recv_second_part = paddle.distributed.irecv(
            tensor=tensor_recv_list[1], src=prev_rank
        )
        ops.append(req_recv_second_part)
    else:
        # Inserted elements are set to be 0.0.
        tensor_recv_list[1] = 0
    if local_rank != len(global_ranks) - 1:
        req_recv_first_part = paddle.distributed.irecv(
            tensor=tensor_recv_list[0], src=next_rank
        )
        ops.append(req_recv_first_part)
        req_send_second_part = paddle.distributed.isend(
            tensor=tensor_send_list[1], dst=next_rank
        )
        ops.append(req_send_second_part)
    else:
        # For the last CP rank, the removed elements of second part go into the first part
        tensor_recv_list[0] = tensor_send_list[1]

    # Wait for all communication operations to complete
    for op in ops:
        op.wait()

    # Splicing: Replace boundary elements with received elements from adjacent ranks
    # This ensures proper sequence continuity across CP boundaries
    index = [slice(None)] * rolled_tensor_list[0].dim()
    index[dims] = shifts
    for i in range(len(rolled_tensor_list)):
        rolled_tensor_list[i][tuple(index)] = tensor_recv_list[i]

    # Concatenate the processed chunks back into a single tensor
    rolled_tensor = paddle.cat(rolled_tensor_list, dim=dims)

    return rolled_tensor, rolled_tensor.sum()


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: paddle.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: paddle.distributed.ProcessGroup | None = None,
        avg_group: paddle.distributed.ProcessGroup | None = None,
    ):
        """Save the mtp loss for logging.
        Args:
            loss (paddle.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
            reduce_group (paddle.distributed.ProcessGroup): The group for reducing the loss.
            mean_group (paddle.distributed.ProcessGroup): The group for averaging the loss.
        """
        # Skip mtp loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = paddle.zeros(
                num_layers, device=paddle.cuda.current_device()
            )
        tracker["values"][layer_number] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    def clean_loss_in_tracker():
        """Clear the mtp losses."""
        tracker = MTPLossLoggingHelper.tracker
        tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    def reduce_loss_in_tracker():
        """Collect and reduce the mtp losses across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]
        # Reduce mtp losses across ranks.
        if tracker.get("reduce_group") is not None:
            paddle.distributed.all_reduce(
                values, group=tracker.get("reduce_group")
            )
        if tracker.get("avg_group") is not None:
            paddle.distributed.all_reduce(
                values,
                group=tracker["avg_group"],
                op=paddle.distributed.ReduceOp.AVG,
            )

    def track_mtp_metrics(
        loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None
    ):
        """Track the Multi-Token Prediction (MTP) metrics for logging."""
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        mtp_losses = tracker["values"] * loss_scale
        mtp_num_layers = mtp_losses.shape[0]
        for i in range(mtp_num_layers):
            name = f"mtp_{i + 1} loss"
            loss = mtp_losses[i]
            if total_loss_dict is not None:
                if name in total_loss_dict:
                    total_loss_dict[name] += loss
                else:
                    total_loss_dict[name] = loss
            if writer is not None:
                writer.add_scalar(name, loss, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"{name}": loss}, iteration)

        MTPLossLoggingHelper.clean_loss_in_tracker()


@dataclass
class MultiTokenPredictionLayerSublayers:
    """
    Dataclass for specifying the sublayers of a MultiTokenPrediction layer.

    Args:
        hnorm (Union[LayerSpec, type]): Specification or instance of the
             hidden states normalization to be applied.
        enorm (Union[LayerSpec, type]): Specification or instance of the
            embedding normalization to be applied.
        eh_proj (Union[LayerSpec, type]): Specification or instance of the
            linear projection to be applied.
        transformer_layer (Union[LayerSpec, type]): Specification
            or instance of the transformer block to be applied.
    """

    enorm: LayerSpec | type = None
    hnorm: LayerSpec | type = None
    eh_proj: LayerSpec | type = None
    transformer_layer: LayerSpec | type = None
    layer_norm: LayerSpec | type = None


def get_mtp_layer_spec(transformer_layer_spec: LayerSpec) -> LayerSpec:
    """Get the MTP layer spec.

    Returns:
        LayerSpec: Layer specification with TE layers
    """
    return get_mtp_layer_spec_for_backend(
        transformer_layer_spec,
        LocalSpecProvider(),
    )


def get_mtp_layer_spec_for_backend(
    transformer_layer_spec: LayerSpec, backend: BackendSpecProvider
) -> LayerSpec:
    """Get the MTP layer spec.

    Returns:
        LayerSpec: Layer specification with layers from the backend.
    """
    column_parallel_linear_impl: type = backend.column_parallel_linear()
    layer_norm_impl: type = backend.layer_norm()
    mtp_layer_spec = LayerSpec(
        layer=MultiTokenPredictionLayer,
        sublayers=MultiTokenPredictionLayerSublayers(
            enorm=layer_norm_impl,
            hnorm=layer_norm_impl,
            eh_proj=column_parallel_linear_impl,
            transformer_layer=transformer_layer_spec,
            layer_norm=layer_norm_impl,
        ),
    )
    return mtp_layer_spec


def get_mtp_layer_offset(config: TransformerConfig) -> int:
    """Get the offset of the MTP layer."""
    # Currently, we only support put all of MTP layers on the last pipeline stage.
    return 0


def get_mtp_num_layers_to_build(
    config: TransformerConfig,
    vp_stage: int | None = None,
    pp_rank: int | None = None,
) -> int:
    """Get the number of MTP layers to build."""
    # Currently, we only support put all of MTP layers on the last pipeline stage.
    vp_size = config.virtual_pipeline_model_parallel_size
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1
    if (
        is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size)
        and is_last_pp_stage
    ):
        return config.mtp_num_layers if config.mtp_num_layers else 0
    else:
        return 0


class MTPLossAutoScaler(paddle.autograd.PyLayer):
    """An AutoScaler that triggers the backward pass and scales the grad for mtp loss."""

    main_loss_backward_scale: paddle.Tensor = paddle.tensor(1.0)

    @staticmethod
    def forward(ctx, output: paddle.Tensor, mtp_loss: paddle.Tensor):
        """Preserve the mtp by storing it in the context to avoid garbage collection.

        Args:
            output (paddle.Tensor): The output tensor.
            mtp_loss (paddle.Tensor): The mtp loss tensor.

        Returns:
            paddle.Tensor: The output tensor.
        """
        ctx.save_for_backward(mtp_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: paddle.Tensor):
        """Compute and scale the gradient for mtp loss..

        Args:
            grad_output (paddle.Tensor): The gradient of the output.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: The gradient of the output, scaled mtp loss
                                               gradient.
        """
        (mtp_loss,) = ctx.saved_tensors
        mtp_loss_backward_scale = MTPLossAutoScaler.main_loss_backward_scale
        scaled_mtp_loss_grad = (
            paddle.ones_like(mtp_loss) * mtp_loss_backward_scale
        )
        return grad_output, scaled_mtp_loss_grad

    @staticmethod
    def set_loss_scale(scale: paddle.Tensor):
        """set the scale of the mtp loss.

        Args:
            scale (paddle.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        MTPLossAutoScaler.main_loss_backward_scale = scale


class MultiTokenPredictionLayer(FleetLayer):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential layers to predict
    D additional tokens.

    The k-th MTP layer consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers: MultiTokenPredictionLayerSublayers,
        layer_number: int = 1,
        vp_stage: int | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)
        self.sequence_parallel = config.sequence_parallel
        self.sublayers = sublayers
        self.layer_number = layer_number
        self.vp_stage = vp_stage
        self.cp_group = pg_collection.cp

        self_attention_spec = (
            self.sublayers.transformer_layer.sublayers.self_attention
        )
        attn_mask_type = self_attention_spec.params.get("attn_mask_type", "")
        assert attn_mask_type in SUPPORTED_ATTN_MASK, (
            "Multi-Token Prediction (MTP) is not jet supported with "
            + f"{attn_mask_type} attention mask type."
            + f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
        )

        self.enorm = build_layer(
            self.sublayers.enorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.hnorm = build_layer(
            self.sublayers.hnorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
        # of the i-th token's hidden states and the (i + K)-th token's decoder input,
        # so the input's shape is [s, b, 2*h].
        # The output will be send to the following transformer layer,
        # so the output's shape should be [s, b, h].
        self.eh_proj = build_layer(
            self.sublayers.eh_proj,
            self.config.hidden_size * 2,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        self.transformer_layer = build_layer(
            self.sublayers.transformer_layer,
            config=self.config,
            vp_stage=vp_stage,
        )

        self.final_layernorm = build_layer(
            self.sublayers.layer_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.offload_context = nullcontext()

    def _get_embeddings(
        self,
        input_ids: paddle.Tensor,
        position_ids: paddle.Tensor,
        embedding: Callable,
        hidden_states: paddle.Tensor,
    ):
        """
        Preprocesses input data for the Multi-Token Prediction (MTP) layers.

        This function computes the decoder input and sends updated input_ids and position_ids to
        the next layer.

        Args:
            input_ids (paddle.Tensor): The input token IDs.
            position_ids (paddle.Tensor): The position IDs corresponding to the input tokens.
            embedding (Callable): The embedding layer
                from gpt model to compute the decoder input.
            hidden_states (paddle.Tensor): hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
        """
        # Calc logits for the current Multi-Token Prediction (MTP) layers.
        input_ids, _ = roll_tensor(
            input_ids, shifts=-1, dims=-1, cp_group=self.cp_group
        )
        position_ids, _ = roll_tensor(
            position_ids, shifts=-1, dims=-1, cp_group=self.cp_group
        )
        # embedding
        decoder_input = embedding(
            input_ids=input_ids, position_ids=position_ids
        )

        return input_ids, position_ids, decoder_input, hidden_states

    def _concat_embeddings(
        self, hidden_states: paddle.Tensor, decoder_input: paddle.Tensor
    ):
        """
        Concatenate the tokens before sending to transformer layer.
        """
        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)
        # At the (k - 1)-th MTP layer, concatenates the i-th token's hidden_states
        # and the (i + K)-th token's embedding, and combine them with linear projection.
        hidden_states = paddle.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel we need to gather the tensor across the model-parallel
        # ranks after the linear projection. This used to call
        # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
        # the gradient in backward pass and was therefore incorrect in this context.
        # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
        hidden_states = gather_from_tensor_model_parallel_region(hidden_states)
        # For sequence parallel, scatter after linear_fc and before transformer layer.
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        return hidden_states

    def _proj_and_transformer_layer(
        self,
        hidden_states: Tensor,
        decoder_input: Tensor,
        attention_mask: paddle.Tensor | None = None,
        context: paddle.Tensor | None = None,
        context_mask: paddle.Tensor | None = None,
        rotary_pos_emb: paddle.Tensor | None = None,
        rotary_pos_cos: paddle.Tensor | None = None,
        rotary_pos_sin: paddle.Tensor | None = None,
        attention_bias: paddle.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> paddle.Tensor:
        """
        Concatenates embeddings with hidden states and then applies transformer layer forward.
        """
        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            hidden_states = self._concat_embeddings(
                hidden_states, decoder_input
            )

            hidden_states, _ = self.transformer_layer(
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

        hidden_states = self._postprocess(hidden_states)

        return hidden_states

    def _postprocess(self, hidden_states: paddle.Tensor):
        """
        Postprocesses the output of the transformer layers.
        """

        # Layer norm before shared head layer.
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states

    def _checkpointed_forward(self, forward_func, *args, **kwargs):
        def checkpoint_handler():
            """Determines whether to use the `tensor_parallel.checkpoint`"""
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                *args,
                *kwargs.values(),
            )

        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert self.config.recompute_num_layers == 1, (
                "recompute_num_layers must be 1 for MTP recompute"
            )
            outputs = checkpoint_handler()
        elif self.config.recompute_method == "block":
            warnings.warn(
                "recompute_method == 'block' is not supported for MTP yet."
                " Skipping recompute."
            )
            outputs = forward_func(*args, **kwargs)
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        embedding=None,
    ):
        """
        Execute the forward pass through the Multi-Token Prediction (MTP) layer.

        Args:
            input_ids (Tensor): Input token IDs .
            position_ids (Tensor): Positional IDs of the input tokens.
            hidden_states (Tensor): Hidden states tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention, if applicable.
            context_mask (Tensor, optional): Mask for cross-attention context, if applicable.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Tensor, optional): Cosine component of rotary positional embeddings.
            rotary_pos_sin (Tensor, optional): Sine component of rotary positional embeddings.
            embedding (Callable): The embedding layer from gpt model to compute the decoder input.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """
        assert context is None, (
            "multi token prediction + cross attention is not yet supported."
        )
        assert packed_seq_params is None, (
            "multi token prediction + sequence packing is not yet supported."
        )

        input_ids, position_ids, decoder_input, hidden_states = (
            self._get_embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                embedding=embedding,
                hidden_states=hidden_states,
            )
        )

        if self.config.recompute_granularity == "full" and self.training:
            hidden_states = self._checkpointed_forward(
                self._proj_and_transformer_layer,
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            hidden_states = self._proj_and_transformer_layer(
                hidden_states=hidden_states,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        return hidden_states, input_ids, position_ids


@dataclass
class MultiTokenPredictionBlockSublayers:
    """
    Dataclass for specifying the sublayers of a multi token prediction block.

    This class defines the structure for configuring the layers, allowing for
    flexible and customizable architecture designs.

    Args:
        layer_specs (list[LayerSpec], optional): A list of layer specifications for
            the layers within the multi token prediction block. Each specification typically
            defines a complete multi token prediction layer (e.g., shared embedding,
            projection matrix, transformer block, shared output head).
    """

    layer_specs: list[LayerSpec] = None


def _get_mtp_block_sublayers(
    config: TransformerConfig,
    spec: MultiTokenPredictionBlockSublayers | LayerSpec,
) -> MultiTokenPredictionBlockSublayers:
    """
    Retrieve or construct MultiTokenPredictionBlockSublayers based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (MultiTokenPredictionBlockSublayers | LayerSpec): Specification for the
            multi token prediction block sublayers.
            Can be either a MultiTokenPredictionBlockSublayers instance or a LayerSpec.

    Returns:
        MultiTokenPredictionBlockSublayers: The sublayers for the multi token prediction block.
    """

    # Transformer block sublayers.
    if isinstance(spec, MultiTokenPredictionBlockSublayers):
        return spec
    elif isinstance(spec, LayerSpec):
        if issubclass(spec.layer, MultiTokenPredictionBlock):
            return spec.sublayers
        else:
            raise Exception(f"specialize for {spec.layer.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class MultiTokenPredictionBlock(FleetLayer):
    """The implementation for Multi-Token Prediction (MTP) which extends
    the prediction scope to multiple future tokens at each position.

    This MTP implementation sequentially predict additional tokens and keep the complete
    causal chain at each prediction depth, by using D sequential layers to predict
    D additional tokens.

    The k-th MTP layer consists of a shared embedding layer, a projection matrix,
    a Transformer block, and a shared output head.

    For the i-th input token at the (k - 1)-th prediction depth, we first combine
    the representation of the i-th token and the embedding of the (i + K)-th token with
    the linear projection. The combined serves as the input of the Transformer block at
    the k-th depth to produce the output representation.

    for more information, please refer to DeepSeek-V3 Technical Report
    https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
    """

    def __init__(
        self,
        config: TransformerConfig,
        spec: TransformerBlockSublayers | LayerSpec,
        vp_stage: int | None = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(config=config)
        self.sublayers = _get_mtp_block_sublayers(config, spec)
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor
        self.vp_stage = vp_stage

        # Initialize Context Parallelism (CP) support for MTP
        # This enables MTP to work with CP > 1 by providing the CP process group
        # to the roll_tensor function for proper boundary communication
        if pg_collection is None:
            # Use default MPU process groups if not provided
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["cp"]
            )
        else:
            # Ensure the provided process groups include CP
            assert hasattr(pg_collection, "cp"), (
                "MultiTokenPredictionBlock pg_collection must have cp process group"
            )

        self._build_layers(pg_collection)
        assert len(self.layers) > 0, (
            "MultiTokenPredictionBlock must have at least one layer."
        )
        self.cp_group = pg_collection.cp

    def _build_layers(self, pg_collection):
        def _build_layer(layer_spec, layer_number):
            layer = build_layer(
                layer_spec,
                config=self.config,
                layer_number=layer_number,
                vp_stage=self.vp_stage,
                pg_collection=pg_collection,
            )
            return layer

        self.layers = paddle.nn.LayerList(
            [
                _build_layer(layer_spec, i + 1)
                for i, layer_spec in enumerate(self.sublayers.layer_specs)
            ]
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        extra_block_kwargs: dict | None = None,
        embedding=None,
    ) -> Tensor:
        """
        Perform the forward pass through all of the MTP layers.

        Args:
            hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
                where s is the sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.

        Returns:
            (Tensor): The mtp loss tensor of shape [b, s].
        """
        # get hidden states from previous mtp stages
        offset = get_mtp_layer_offset(self.config)
        hidden_states_list = list(
            paddle.chunk(hidden_states, 1 + offset, dim=0)
        )
        hidden_states = hidden_states_list[offset]
        for layer_number in range(len(self.layers)):
            (hidden_states, input_ids, position_ids) = self.layers[
                layer_number
            ](
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                embedding=embedding,
                **(extra_block_kwargs or {}),
            )

            # append the output hidden states of the current mtp layer
            # to the hidden_states_list
            hidden_states_list.append(hidden_states)

        # concat the hidden states of all mtp layers
        hidden_states = paddle.cat(hidden_states_list, dim=0)
        return hidden_states
