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
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    ScheduleNode,
    build_spec_layer,
)
from paddle.distributed.fleet.utils import recompute

from paddlefleet import tensor_parallel
from paddlefleet.context_parallel_utils import ContextParallelScatterOp
from paddlefleet.parallel_state import (
    get_context_parallel_world_size,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from paddlefleet.transformer.enums import AttnMaskType
from paddlefleet.transformer.layer import FleetLayer

if TYPE_CHECKING:
    from paddlefleet.models.backends import BackendSpecProvider
    from paddlefleet.packed_seq_params import PackedSeqParams
    from paddlefleet.transformer.transformer_config import TransformerConfig

SUPPORTED_ATTN_MASK = [
    AttnMaskType.padding,
    AttnMaskType.causal,
    AttnMaskType.no_mask,
    AttnMaskType.padding_causal,
]


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: paddle.Tensor,
        layer_number: int,
        num_hidden_layers: int,
        reduce_group: paddle.distributed.communication.group.Group
        | None = None,
        avg_group: paddle.distributed.communication.group.Group | None = None,
    ):
        """Save the mtp loss for logging.
        Args:
            loss (paddle.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_hidden_layers (int): The number of total layers.
            reduce_group (paddle.distributed.communication.group.Group): The group for reducing the loss.
            mean_group (paddle.distributed.communication.group.Group): The group for averaging the loss.
        """
        # Skip mtp loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = paddle.zeros(num_hidden_layers)
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
        num_nextn_predict_layers = mtp_losses.shape[0]
        for i in range(num_nextn_predict_layers):
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
class MultiTokenPredictionLayerSublayersSpec:
    """
    Dataclass for specifying the sublayers_spec of a MultiTokenPrediction layer.

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


def get_mtp_layer_spec_for_backend(
    config: TransformerConfig,
    transformer_layer_spec: LayerSpec,
    backend: BackendSpecProvider,
    layer_number: int,
) -> LayerSpec:
    """Get the MTP layer spec.

    Returns:
        LayerSpec: Layer specification with layers from the backend.
    """
    column_parallel_linear_impl: type = backend.column_parallel_linear()
    layer_norm_impl: type = backend.layer_norm()
    mtp_layer_spec = LayerSpec(
        layer=WeightOnlyMTPLayer
        if config.mtp_load_weight_only
        else MultiTokenPredictionLayer,
        sublayers_spec=MultiTokenPredictionLayerSublayersSpec(
            enorm=layer_norm_impl,
            hnorm=layer_norm_impl,
            eh_proj=column_parallel_linear_impl,
            transformer_layer=transformer_layer_spec,
            layer_norm=layer_norm_impl,
        ),
        extra_kwargs={
            "config": config,
            "layer_number": layer_number,
        },
    )
    return mtp_layer_spec


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
        (mtp_loss,) = ctx.saved_tensor()
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
        sublayers_spec: MultiTokenPredictionLayerSublayersSpec,
        layer_number: int = 1,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.sequence_parallel = config.sequence_parallel
        self.tensor_parallel = config.tensor_model_parallel_size
        self.sublayers_spec = sublayers_spec
        self.layer_number = layer_number
        self.cp_group = pg_collection.cp

        self_attention_spec = (
            self.sublayers_spec.transformer_layer.sublayers_spec.self_attn
        )
        attn_mask_type = self_attention_spec.extra_kwargs.get(
            "attn_mask_type", ""
        )
        assert attn_mask_type in SUPPORTED_ATTN_MASK, (
            "Multi-Token Prediction (MTP) is not jet supported with "
            + f"{attn_mask_type} attention mask type."
            + f"The supported attention mask types are {SUPPORTED_ATTN_MASK}."
        )

        self.enorm = build_spec_layer(
            self.sublayers_spec.enorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        self.hnorm = build_spec_layer(
            self.sublayers_spec.hnorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
        # of the i-th token's hidden states and the (i + K)-th token's decoder input,
        # so the input's shape is [s, b, 2*h].
        # The output will be sent to the following transformer layer,
        # so the output's shape should be [s, b, h].
        self.eh_proj = build_spec_layer(
            self.sublayers_spec.eh_proj,
            self.config.hidden_size * 2,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        self.transformer_layer = build_spec_layer(
            self.sublayers_spec.transformer_layer,
            config=self.config,
        )
        if not self.config.gpt_model_use_experimental_version:
            self.norm = build_spec_layer(
                self.sublayers_spec.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.rms_norm_eps,
            )

        self.offload_context = nullcontext()

    def _concat_embeddings(
        self,
        hidden_states: paddle.Tensor,
        decoder_input: paddle.Tensor,
        mtp_hidden_inputs_mask: paddle.Tensor | None = None,
    ):
        """
        Concatenate the tokens before sending to transformer layer.
        """
        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)
        # Apply mtp_hidden_inputs_mask to mask out hidden state contributions
        # at specific positions (e.g. EOS boundaries) in MTP.
        # mask shape: [B, 1, S] -> [B, S, 1] to broadcast with hidden_states [B, S, H]
        if mtp_hidden_inputs_mask is not None:
            mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.transpose([0, 2, 1])
            mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.astype(
                hidden_states.dtype
            )

            if (
                get_context_parallel_world_size() > 1
                and self.config.experimental_dataflow
            ):
                # In EB dataflow and CP size > 1, mtp_hidden_inputs_mask is [b, s, 1];
                # we need to scatter it to [b, s/cp, 1] here.
                mtp_hidden_inputs_mask = ContextParallelScatterOp.apply(
                    mtp_hidden_inputs_mask, axis=1
                )

            hidden_states = hidden_states * mtp_hidden_inputs_mask
        # At the (k - 1)-th MTP layer, concatenates the i-th token's hidden_states
        # and the (i + K)-th token's embedding, and combine them with linear projection.
        hidden_states = paddle.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel we need to gather the tensor across the model-parallel
        # ranks after the linear projection. This used to call
        # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
        # the gradient in backward pass and was therefore incorrect in this context.
        # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
        if self.tensor_parallel > 1:
            hidden_states = gather_from_tensor_model_parallel_region(
                hidden_states
            )
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
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        mtp_hidden_inputs_mask: paddle.Tensor | None = None,
        input_ids: paddle.Tensor | None = None,
        **kwargs,
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
                hidden_states, decoder_input, mtp_hidden_inputs_mask
            )

            input_dict = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "context": context,
                "context_mask": context_mask,
                "rotary_pos_emb": rotary_pos_emb,
                "rotary_pos_cos": rotary_pos_cos,
                "rotary_pos_sin": rotary_pos_sin,
                "attention_bias": attention_bias,
                "packed_seq_params": packed_seq_params,
                "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
                "is_mtp": True,
                "input_ids": input_ids,
            }
            rst_dict = self.transformer_layer(input_dict)
        if not self.config.gpt_model_use_experimental_version:
            hidden_states = self.norm(rst_dict["hidden_states"])
        else:
            hidden_states = rst_dict["hidden_states"]
        return hidden_states

    def _checkpointed_forward(self, forward_func, *args, **kwargs):
        def checkpoint_handler():
            """Determines whether to use the `tensor_parallel.checkpoint`"""
            hidden_states = kwargs.get("hidden_states", None)
            decoder_input = kwargs.get("decoder_input", None)
            attention_mask = kwargs.get("attention_mask", None)
            attn_mask_startend_row_indices = kwargs.get(
                "attn_mask_startend_row_indices", None
            )
            context = kwargs.get("context", None)
            context_mask = kwargs.get("context_mask", None)
            rotary_pos_emb = kwargs.get("rotary_pos_emb", None)
            rotary_pos_cos = kwargs.get("rotary_pos_cos", None)
            rotary_pos_sin = kwargs.get("rotary_pos_sin", None)
            attention_bias = kwargs.get("attention_bias", None)
            packed_seq_params = kwargs.get("packed_seq_params", None)
            mtp_hidden_inputs_mask = kwargs.get("mtp_hidden_inputs_mask", None)
            input_ids = kwargs.get("input_ids", None)
            return recompute(
                forward_func,
                hidden_states=hidden_states
                if hidden_states is not None
                else None,
                decoder_input=decoder_input
                if decoder_input is not None
                else None,
                attention_mask=attention_mask
                if attention_mask is not None
                else None,
                attn_mask_startend_row_indices=attn_mask_startend_row_indices
                if attn_mask_startend_row_indices is not None
                else None,
                context=context if context is not None else None,
                context_mask=context_mask if context_mask is not None else None,
                rotary_pos_emb=rotary_pos_emb
                if rotary_pos_emb is not None
                else None,
                rotary_pos_cos=rotary_pos_cos
                if rotary_pos_cos is not None
                else None,
                rotary_pos_sin=rotary_pos_sin
                if rotary_pos_sin is not None
                else None,
                attention_bias=attention_bias
                if attention_bias is not None
                else None,
                packed_seq_params=packed_seq_params
                if packed_seq_params is not None
                else None,
                mtp_hidden_inputs_mask=mtp_hidden_inputs_mask
                if mtp_hidden_inputs_mask is not None
                else None,
                input_ids=input_ids if input_ids is not None else None,
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

    def forward(self, dict_args: dict):
        if "context" in dict_args:
            assert dict_args["context"] is None, (
                "multi token prediction + cross attention is not yet supported."
            )
        if "packed_seq_params" in dict_args:
            assert dict_args["packed_seq_params"] is None, (
                "multi token prediction + sequence packing is not yet supported."
            )

        hidden_states_concat = dict_args["hidden_states"]
        # New dataflow: pop mtp_startend_row_indices_all if present (experimental_dataflow=True)
        # Shape: [B, num_nextn_predict_layers, S, 1]
        origin_start_row_indices = dict_args.pop(
            "attn_mask_startend_row_indices", None
        )
        mtp_startend_row_indices_all = dict_args.pop(
            "mtp_startend_row_indices_all", None
        )
        mtp_hidden_inputs_mask_all = dict_args.pop(
            "mtp_hidden_inputs_mask_all", None
        )
        # Pop per-depth MTP input_ids for MoE routing mask.
        # Shape: [B, num_nextn_predict_layers, max_seq] when present, None otherwise.
        mtp_input_ids_for_moe_mask = dict_args.pop(
            "mtp_input_ids_for_moe_mask", None
        )
        # Save and clear backbone input_ids so it doesn't leak into MTP transformer layers
        origin_input_ids = dict_args.pop("input_ids", None)

        # Trim rotary_pos_emb to main decoder length (remove MTP extra positions)
        # rotary_pos_emb includes extra positions beyond the main decoder length;
        # MTP's internal transformer_layer processes main-length sequences only.
        # Compute main_seq_len from the split hidden_states shape.
        n = self.config.num_nextn_predict_layers
        if self.config.sequence_parallel:
            main_seq_len = (
                hidden_states_concat.shape[0]
                // (n + 1)
                * self.config.tensor_model_parallel_size
            )
        else:
            # Non-SP: MTP parts are concatenated on batch dim (axis=0),
            # so shape[1] is already the per-part sequence length.
            main_seq_len = hidden_states_concat.shape[1]
        origin_rotary_pos_emb = dict_args.get("rotary_pos_emb", None)
        if origin_rotary_pos_emb is not None:
            if self.config.sequence_parallel:
                dict_args["rotary_pos_emb"] = origin_rotary_pos_emb[
                    :main_seq_len
                ]
            else:
                dict_args["rotary_pos_emb"] = origin_rotary_pos_emb[
                    :, :main_seq_len
                ]
        origin_rotary_pos_cos = dict_args.get("rotary_pos_cos", None)
        if origin_rotary_pos_cos is not None:
            dict_args["rotary_pos_cos"] = origin_rotary_pos_cos[
                :, :main_seq_len
            ]
        origin_rotary_pos_sin = dict_args.get("rotary_pos_sin", None)
        if origin_rotary_pos_sin is not None:
            dict_args["rotary_pos_sin"] = origin_rotary_pos_sin[
                :, :main_seq_len
            ]
        # Shape check: mtp_startend_row_indices_all [B, num_nextn, S, 1],
        #              mtp_hidden_inputs_mask_all   [B, num_nextn, S]
        if mtp_startend_row_indices_all is not None:
            num_nextn = self.config.num_nextn_predict_layers
            assert mtp_startend_row_indices_all.shape[1] == num_nextn, (
                f"mtp_startend_row_indices_all.shape[1]={mtp_startend_row_indices_all.shape[1]} "
                f"!= num_nextn_predict_layers={num_nextn}"
            )
        if mtp_hidden_inputs_mask_all is not None:
            num_nextn = self.config.num_nextn_predict_layers
            assert mtp_hidden_inputs_mask_all.shape[1] == num_nextn, (
                f"mtp_hidden_inputs_mask_all.shape[1]={mtp_hidden_inputs_mask_all.shape[1]} "
                f"!= num_nextn_predict_layers={num_nextn}"
            )
        if (
            mtp_startend_row_indices_all is not None
            and mtp_hidden_inputs_mask_all is not None
        ):
            assert mtp_startend_row_indices_all.shape[:3] == [
                mtp_hidden_inputs_mask_all.shape[0],
                mtp_hidden_inputs_mask_all.shape[1],
                mtp_hidden_inputs_mask_all.shape[2],
            ], (
                f"mtp_startend_row_indices_all shape {mtp_startend_row_indices_all.shape} "
                f"and mtp_hidden_inputs_mask_all shape {mtp_hidden_inputs_mask_all.shape} "
                f"mismatch on [B, num_nextn, S] dims"
            )
        if self.config.train_mtp_only:
            for i in range(self.config.num_nextn_predict_layers):
                tensor_list = paddle.split(
                    hidden_states_concat,
                    self.config.num_nextn_predict_layers + 1,
                )
                dict_args["hidden_states"] = tensor_list[i]
                dict_args["decoder_input"] = tensor_list[i + 1]

                # New dataflow: get the mask for depth i, shape [B, 1, S, 1]
                mtp_mask_i = None
                if mtp_startend_row_indices_all is not None:
                    mtp_mask_i = mtp_startend_row_indices_all[
                        :, i : i + 1, :, :
                    ]
                    dict_args["attn_mask_startend_row_indices"] = mtp_mask_i

                # New dataflow: get hidden inputs mask for depth i, shape [B, 1, S]
                if mtp_hidden_inputs_mask_all is not None:
                    dict_args["mtp_hidden_inputs_mask"] = (
                        mtp_hidden_inputs_mask_all[:, i : i + 1, :]
                    )

                # Get per-depth input_ids for MoE routing mask
                if mtp_input_ids_for_moe_mask is not None:
                    dict_args["input_ids"] = mtp_input_ids_for_moe_mask[
                        :, i, :
                    ].contiguous()
                else:
                    dict_args.pop("input_ids", None)

                if (
                    self.config.recompute_granularity == "full"
                    and self.training
                ):
                    hidden_states = self._checkpointed_forward(
                        self._proj_and_transformer_layer,
                        **dict_args,
                    )
                else:
                    hidden_states = self._proj_and_transformer_layer(
                        **dict_args,
                    )

                tensor_list[i + 1] = hidden_states
                hidden_states_concat = paddle.concat(tensor_list)
            dict_args["hidden_states"] = hidden_states_concat
            dict_args.pop("decoder_input")
        else:
            tensor_list = paddle.split(
                hidden_states_concat, self.config.num_nextn_predict_layers + 1
            )
            dict_args["hidden_states"] = tensor_list[self.layer_number]
            dict_args["decoder_input"] = tensor_list[self.layer_number + 1]

            # New dataflow: get the mask for this layer's depth, shape [B, 1, S, 1]
            mtp_mask = None
            if mtp_startend_row_indices_all is not None:
                if self.config.gpt_model_use_experimental_version:
                    mtp_mask = mtp_startend_row_indices_all[
                        :,
                        self.layer_number : self.layer_number + 1,
                        :,
                        :,
                    ]
                else:
                    mtp_mask = mtp_startend_row_indices_all[
                        :,
                        self.layer_number : self.layer_number + 1,
                        :,
                        :1,
                    ]
                dict_args["attn_mask_startend_row_indices"] = mtp_mask

            # New dataflow: get hidden inputs mask for this layer's depth, shape [B, 1, S]
            if mtp_hidden_inputs_mask_all is not None:
                dict_args["mtp_hidden_inputs_mask"] = (
                    mtp_hidden_inputs_mask_all[
                        :, self.layer_number : self.layer_number + 1, :
                    ]
                )

            # Get per-depth input_ids for MoE routing mask
            if mtp_input_ids_for_moe_mask is not None:
                dict_args["input_ids"] = mtp_input_ids_for_moe_mask[
                    :, self.layer_number, :
                ].contiguous()
            else:
                dict_args.pop("input_ids", None)

            # print(dict_args["attn_mask_startend_row_indices"])
            # assert 0
            if self.config.recompute_granularity == "full" and self.training:
                hidden_states = self._checkpointed_forward(
                    self._proj_and_transformer_layer,
                    **dict_args,
                )
            else:
                hidden_states = self._proj_and_transformer_layer(
                    **dict_args,
                )

            tensor_list[self.layer_number + 1] = hidden_states
            hidden_states_concat = paddle.concat(tensor_list)
            dict_args["hidden_states"] = hidden_states_concat
            dict_args.pop("decoder_input")

        # Restore mtp_startend_row_indices_all for subsequent MTP layers (num_nextn > 1)
        if mtp_startend_row_indices_all is not None:
            dict_args["mtp_startend_row_indices_all"] = (
                mtp_startend_row_indices_all
            )
        # Restore mtp_hidden_inputs_mask_all for subsequent MTP layers (num_nextn > 1)
        if mtp_hidden_inputs_mask_all is not None:
            dict_args["mtp_hidden_inputs_mask_all"] = mtp_hidden_inputs_mask_all
        # Restore mtp_input_ids_for_moe_mask for subsequent MTP layers (num_nextn > 1)
        if mtp_input_ids_for_moe_mask is not None:
            dict_args["mtp_input_ids_for_moe_mask"] = mtp_input_ids_for_moe_mask
        # Restore backbone input_ids
        if origin_input_ids is not None:
            dict_args["input_ids"] = origin_input_ids
        else:
            dict_args.pop("input_ids", None)
        # Restore rotary_pos_emb/cos/sin to full length
        if origin_rotary_pos_emb is not None:
            dict_args["rotary_pos_emb"] = origin_rotary_pos_emb
        if origin_rotary_pos_cos is not None:
            dict_args["rotary_pos_cos"] = origin_rotary_pos_cos
        if origin_rotary_pos_sin is not None:
            dict_args["rotary_pos_sin"] = origin_rotary_pos_sin
        # Clean up per-depth slice key
        dict_args.pop("mtp_hidden_inputs_mask", None)
        if origin_start_row_indices is not None:
            dict_args["attn_mask_startend_row_indices"] = (
                origin_start_row_indices
            )
        return dict_args

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="MultiTokenPredictionLayer")


class WeightOnlyMTPLayer(MultiTokenPredictionLayer):
    """MTP layer that only holds weights without participating in forward computation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for _, param in self.state_dict().items():
            param.is_weight_only_mtp = True

    def forward(self, dict_args: dict):
        return dict_args

    def build_schedule_node(self):
        return ScheduleNode(self.forward, name="WeightOnlyMTPLayer")
