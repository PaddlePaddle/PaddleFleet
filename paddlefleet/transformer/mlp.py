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
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F

# (TODO): need adapt to flex_checkpoint
# dist_checkpoint in paddle is flex_checkpoint which have many difference.
# from paddlefleet.dist_checkpointing import ShardedTensor
# from paddlefleet.dist_checkpointing.mapping import (
#     ReplicaId,
#     ShardedStateDict,
#     ShardedTensorFactory,
# )
from paddlefleet.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from paddlefleet.fusions.fused_bias_gelu import bias_gelu_impl
from paddlefleet.fusions.fused_bias_swiglu import (
    bias_swiglu_impl,
    weighted_bias_swiglu_impl,
)
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.spec_utils import LayerSpec, build_layer

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)

logger = logging.getLogger(__name__)


# pylint: disable=missing-class-docstring
@dataclass
class MLPSublayersSpec:
    """
    The dataclass for LayerSpecs of MLP sublayers_spec
    including  linear fc1, activation function, linear fc2.
    """

    linear_fc1: LayerSpec | type = None
    activation_func: LayerSpec | type = None
    linear_fc2: LayerSpec | type = None


class MLP(FleetLayer):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: MLPSublayersSpec,
        is_expert: bool = False,
        input_size: int | None = None,
        ffn_hidden_size: int | None = None,
        tp_group=None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = (
            input_size if input_size is not None else self.config.hidden_size
        )

        tp_group = get_tensor_model_parallel_group_if_none(
            tp_group, is_expert=is_expert
        )
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError(
                    "MoE MLP requires `ffn_hidden_size`, but it was not provided."
                )
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = self.config.ffn_hidden_size

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2
        print("sublayers_spec", sublayers_spec, sublayers_spec.linear_fc1)
        self.linear_fc1 = build_layer(
            sublayers_spec.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_group=tp_group,
        )

        self.activation_func = self.config.activation_func

        self.linear_fc2 = build_layer(
            sublayers_spec.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_group=tp_group,
        )

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if (
                    self.activation_func == F.silu
                    and self.config.gated_linear_unit
                ):
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif (
                    self.activation_func == quick_gelu
                    and self.config.gated_linear_unit
                ):
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(
                            intermediate_parallel, bias_parallel
                        )
                elif (
                    self.activation_func == F.silu
                    and self.config.gated_linear_unit
                ):
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and False,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x_glu, x_linear = paddle.chunk(x, 2, axis=-1)
                    if (
                        val := self.config.activation_func_clamp_value
                    ) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(
                    intermediate_parallel
                )

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = (
                    intermediate_parallel * per_token_scale.unsqueeze(-1)
                )
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")
        output, output_bias = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias

    # (TODO): need to adapt flex_checkpoint logic
    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ):
        """Return the sharded state dictionary of the module."""
        pass

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


# (TODO): need to adapt flex_checkpoint logic
# pylint: disable=missing-function-docstring
def apply_swiglu_sharded_factory(
    original_sh_ten, sharded_offsets, singleton_local_shards: bool = False
):
    # We must split the tensor into 2 parts, each sharded separately.
    # This requires a ShardedTensorFactory which `chunk`s during saving
    # and `cat`s during loading
    pass
