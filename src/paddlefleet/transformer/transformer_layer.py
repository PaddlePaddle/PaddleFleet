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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn

from paddlefleet import tensor_parallel
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.spec_utils import LayerSpec, build_layer
from paddlefleet.transformer.identity_op import IdentityFuncOp, IdentityOp
from paddlefleet.transformer.mlp import MLP
from paddlefleet.utils import log_single_rank

if TYPE_CHECKING:
    from paddlefleet.packed_seq_params import PackedSeqParams
    from paddlefleet.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformerLayerSublayersSpec:
    """
    Configuration class for specifying the sublayers_spec of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm (LayerSpec | type): Specification for the input layer normalization.
        self_attn (LayerSpec | type): Specification for the self-attention mechanism.
        self_attn_bda (LayerSpec | type): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm (LayerSpec | type): Specification for the layer
            normalization before cross-attention.
        cross_attention (LayerSpec | type): Specification for the cross-attention mechanism.
        cross_attn_bda (LayerSpec | type): Specification for the bias-dropout-add operation
            after cross-attention.
        post_attention_layernorm (LayerSpec | type): Specification for the layer normalization
            before the MLP.
        mlp (LayerSpec | type): Specification for the MLP in Dense layer.
        mlp_bda (LayerSpec | type): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: LayerSpec | type = IdentityOp
    self_attn: LayerSpec | type = IdentityOp
    self_attn_bda: LayerSpec | type = IdentityFuncOp

    pre_cross_attn_layernorm: LayerSpec | type = IdentityOp
    cross_attention: LayerSpec | type = IdentityOp
    cross_attn_bda: LayerSpec | type = IdentityFuncOp

    post_attention_layernorm: LayerSpec | type = IdentityOp
    mlp: LayerSpec | type = IdentityOp
    mlp_bda: LayerSpec | type = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: dict[str, str] = field(default_factory=dict)


class TransformerLayer(nn.Layer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: TransformerLayerSublayersSpec,
        layer_number: int = 1,
        hidden_dropout_prob: float | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ):
        super().__init__()

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.config = config

        self.layer_number = layer_number
        self.hidden_dropout_prob = (
            config.hidden_dropout_prob
            if hidden_dropout_prob is None
            else hidden_dropout_prob
        )

        # [Layer 1: Input Layernorm] Optional Layernorm on the input data
        self.input_layernorm = build_layer(
            sublayers_spec.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[
                    self.layer_number
                ]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        attention_optional_kwargs["pg_collection"] = pg_collection

        # [Layer 2: SelfAttention]
        self.self_attn = build_layer(
            sublayers_spec.self_attn,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Layer 3: BiasDropoutFusion]
        self.self_attn_bda = build_layer(sublayers_spec.self_attn_bda)

        # [Layer 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = build_layer(
            sublayers_spec.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        # [Layer 5: CrossAttention]
        self.cross_attention = build_layer(
            sublayers_spec.cross_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
        )

        # [Layer 6: BiasDropoutFusion]
        self.cross_attn_bda = build_layer(
            sublayers_spec.cross_attn_bda, config=self.config
        )

        # [Layer 7: Pre MLP] Optional Layernorm before MLP
        self.post_attention_layernorm = build_layer(
            sublayers_spec.post_attention_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        # [Layer 8: MLP block]
        additional_mlp_kwargs = {}

        from paddlefleet.transformer.moe.moe_layer import MoELayer

        # MLP expects tp_group but MoELayer expects pg_collection to be passed in.
        # We can change MLP to accept pg_collection but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if sublayers_spec.mlp is not a LayerSpec,we dont have to handle passing additional kwargs
        if isinstance(sublayers_spec.mlp, LayerSpec):
            if sublayers_spec.mlp.layer == MoELayer:
                additional_mlp_kwargs["pg_collection"] = pg_collection
            elif sublayers_spec.mlp.layer == MLP:
                assert hasattr(pg_collection, "tp"), (
                    "TP process group is required for MLP in TransformerLayer"
                )
                additional_mlp_kwargs["tp_group"] = pg_collection.tp
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unknown MLP type: {type(sublayers_spec.mlp)}. Using default kwargs.",
                )

        self.mlp = build_layer(
            sublayers_spec.mlp, config=self.config, **additional_mlp_kwargs
        )
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)

        # [Layer 9: BiasDropoutFusion]
        self.mlp_bda = build_layer(sublayers_spec.mlp_bda)

        self.recompute_input_layernorm = False
        self.recompute_post_attention_layernorm = False
        self.recompute_mlp = False
        if self.config.recompute_granularity == "selective":
            if "layernorm" in self.config.recompute_layers:
                if not isinstance(self.post_attention_layernorm, IdentityOp):
                    self.recompute_post_attention_layernorm = True

            if "mlp" in self.config.recompute_layers:
                if not isinstance(self.mlp, MoELayer):
                    self.recompute_mlp = True

    def forward(
        self,
        dict_args: dict,
    ):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        # Remove 'dynamic_inference_decode_only' from kwargs if present
        # this is only used to uniquely identify decode and non-decode cuda graph
        # runners in the cuda graph manager
        dict_args.pop("dynamic_inference_decode_only", None)
        hidden_states, context = self._forward_attention(**dict_args)
        output = self._forward_mlp(hidden_states)
        rst = {"hidden_states": output}
        if context is not None:
            rst["context"] = context
        rst = {**dict_args, **rst}
        return rst

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        context: Tensor | None = None,
        context_mask: Tensor | None = None,
        rotary_pos_emb: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor | None): Mask tensor for self-attention.
            context (Tensor | None): Context tensor for cross-attention.
            context_mask (Tensor | None): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor | None): Rotary positional embeddings.
            attention_bias (Tensor | None): Bias tensor for Q * K.T.
            packed_seq_params (object, optional): Parameters for packed sequence processing.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        if self.recompute_input_layernorm:
            self.input_layernorm_checkpoint = (
                tensor_parallel.CheckpointWithoutOutput()
            )
            input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
                self.input_layernorm, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attn(
            input_layernorm_output,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute
            # as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        with paddle.enable_grad():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout_prob)

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(
            hidden_states
        )

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
        )

        if (
            isinstance(attention_output_with_bias, dict)
            and "context" in attention_output_with_bias
        ):
            context = attention_output_with_bias["context"]

        with paddle.enable_grad():
            hidden_states = self.cross_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout_prob)

        return hidden_states, context

    def _forward_mlp(self, hidden_states):
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        if self.recompute_post_attention_layernorm:
            self.pre_mlp_norm_checkpoint = (
                tensor_parallel.CheckpointWithoutOutput()
            )
            post_attention_layernorm_output = (
                self.pre_mlp_norm_checkpoint.checkpoint(
                    self.post_attention_layernorm, hidden_states
                )
            )
        else:
            post_attention_layernorm_output = self.post_attention_layernorm(
                hidden_states
            )

        if self.recompute_mlp:
            mlp_output_with_bias = tensor_parallel.checkpoint(
                self.mlp, False, post_attention_layernorm_output
            )
        else:
            mlp_output_with_bias = self.mlp(post_attention_layernorm_output)

        if self.recompute_post_attention_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(
                mlp_output_with_bias[0]
            )

        with paddle.enable_grad():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.hidden_dropout_prob)

        return hidden_states
