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
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

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

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)

logger = logging.getLogger(__name__)

_ACCURACY_COMPATIBLE_KERNEL = (
    os.environ.get("FLAGS_use_accuracy_compatible_kernel", "0") == "1"
)


def _accuracy_compatible_swiglu(hidden_states):
    gate, linear = paddle.chunk(hidden_states, 2, axis=-1)
    return F.silu(gate) * linear


class _AccuracyCompatibleRouterScaleGradFunction(paddle.autograd.PyLayer):
    """Apply router scale while matching Megatron's grouped reduction shape."""

    @staticmethod
    def forward(ctx, activation, scale, reduction_rows):
        ctx.activation_dtype = activation.dtype
        ctx.reduction_rows = int(reduction_rows)
        ctx.save_for_backward(activation, scale)
        return (activation * scale.unsqueeze(-1)).cast(activation.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        activation, scale = ctx.saved_tensor()
        row_count = activation.shape[0]
        products = activation.cast("float32") * grad_output.cast("float32")
        if ctx.reduction_rows > row_count:
            products = paddle.concat(
                [
                    products,
                    paddle.zeros(
                        [ctx.reduction_rows - row_count, products.shape[-1]],
                        dtype=products.dtype,
                    ),
                ],
                axis=0,
            )
        grad_scale = products.sum(axis=-1)[:row_count]
        return None, grad_scale.cast(scale.dtype)


def _accuracy_compatible_router_scale(activation, scale, reduction_rows):
    native_activation_path = (
        activation * scale.detach().unsqueeze(-1)
    ).cast(activation.dtype)
    scale_path = _AccuracyCompatibleRouterScaleGradFunction.apply(
        activation.detach(), scale, reduction_rows
    )
    return native_activation_path + (scale_path - scale_path.detach())


class _AccuracyCompatibleLinearInputGradFunction(paddle.autograd.PyLayer):
    """Linear forward with a materialized-transpose input gradient."""

    @staticmethod
    def forward(ctx, hidden_states, weight):
        ctx.save_for_backward(weight)
        return F.linear(hidden_states, weight)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensor()
        grad_input = paddle.matmul(
            grad_output, weight.transpose([1, 0]).contiguous()
        )
        return grad_input, None


def _accuracy_compatible_projection(projection, hidden_states):
    output_bias = projection.bias if projection.skip_bias_add else None
    bias = None if projection.skip_bias_add else projection.bias
    output = _AccuracyCompatibleLinearInputGradFunction.apply(
        hidden_states, projection.weight.detach()
    )
    if bias is not None:
        output = output + bias.detach()
    parameter_path = F.linear(hidden_states.detach(), projection.weight, bias)
    output = output + (parameter_path - parameter_path.detach())
    return output, output_bias


# pylint: disable=missing-class-docstring
@dataclass
class MLPSublayersSpec:
    """
    The dataclass for LayerSpecs of MLP sublayers_spec
    including  linear fc1, activation function, linear fc2.
    """

    up_gate_proj: LayerSpec | type = None
    hidden_act: LayerSpec | type = None
    down_proj: LayerSpec | type = None


class MLP(FleetLayer):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.use_bias is False, the bias returned is None.

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
        intermediate_size: int | None = None,
        hidden_size: int | None = None,
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
        if intermediate_size is None:
            if is_expert:
                raise ValueError(
                    "MoE MLP requires `intermediate_size`, but it was not provided."
                )
            warnings.warn(
                "MLP requires intermediate_size, but it was not provided. Using \
                    config.intermediate_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.config.intermediate_size is None:
                raise ValueError(
                    "MLP requires `config.intermediate_size` is not None, but it got None."
                )

            intermediate_size = self.config.intermediate_size

        self.hidden_size = (
            hidden_size if hidden_size is not None else self.config.hidden_size
        )

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            intermediate_size *= 2
        self.up_gate_proj = build_spec_layer(
            sublayers_spec.up_gate_proj,
            self.input_size,
            intermediate_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.use_bias,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_group=tp_group,
        )

        # Ensure hidden_act is a callable function, not a bound method
        hidden_act_value = self.config.hidden_act
        if hasattr(hidden_act_value, "__self__") and hasattr(
            hidden_act_value, "__func__"
        ):
            # If it's a bound method, use the unbound function
            self.hidden_act = hidden_act_value.__func__
        else:
            self.hidden_act = hidden_act_value

        if self.config.gated_linear_unit:
            intermediate_size //= 2

        self.down_proj = build_spec_layer(
            sublayers_spec.down_proj,
            intermediate_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.use_bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_group=tp_group,
        )

    def forward(
        self,
        hidden_states,
        per_token_scale=None,
        accuracy_compatible_router_reduction_rows=None,
    ):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="up_gate_proj")
        if (
            _ACCURACY_COMPATIBLE_KERNEL
            and self.config.tensor_model_parallel_size == 1
        ):
            intermediate_parallel, bias_parallel = _accuracy_compatible_projection(
                self.up_gate_proj, hidden_states
            )
        else:
            intermediate_parallel, bias_parallel = self.up_gate_proj(hidden_states)
        nvtx_range_pop(suffix="up_gate_proj")

        nvtx_range_push(suffix="activation")

        # Alignment mode: use Paddle native F.swiglu
        _use_paddle_swiglu = getattr(
            self.config, "gpt_model_use_experimental_version", False
        )
        if (
            _ACCURACY_COMPATIBLE_KERNEL
            and bias_parallel is None
            and self.hidden_act == F.silu
            and self.config.gated_linear_unit
        ):
            intermediate_parallel = _accuracy_compatible_swiglu(
                intermediate_parallel
            )
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                if accuracy_compatible_router_reduction_rows is not None:
                    intermediate_parallel = _accuracy_compatible_router_scale(
                        intermediate_parallel,
                        per_token_scale,
                        accuracy_compatible_router_reduction_rows,
                    )
                else:
                    intermediate_parallel = (
                        intermediate_parallel * per_token_scale.unsqueeze(-1)
                    ).to(original_dtype)
        elif (
            self.config.use_bias
            and self.config.gpt_model_use_experimental_version
            and self.config.tensor_model_parallel_size == 1
        ):
            hidden_states = paddle.incubate.nn.functional.fused_linear(
                hidden_states, self.up_gate_proj.weight, self.up_gate_proj.bias
            )
            hidden_states = F.swiglu(hidden_states)
            output = paddle.incubate.nn.functional.fused_linear(
                hidden_states, self.down_proj.weight, self.down_proj.bias
            )
            return output, None

        elif (
            _use_paddle_swiglu
            and self.hidden_act == F.silu
            and self.config.gated_linear_unit
        ):
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = F.swiglu(intermediate_parallel)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.hidden_act == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        getattr(
                            self.config,
                            "activation_func_fp8_input_store",
                            False,
                        ),
                        self.config.activation_func_clamp_value,
                    )
                elif (
                    self.hidden_act == quick_gelu
                    and self.config.gated_linear_unit
                ):
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        getattr(
                            self.config,
                            "activation_func_fp8_input_store",
                            False,
                        ),
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.hidden_act == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.use_bias is True
                        intermediate_parallel = bias_gelu_impl(
                            intermediate_parallel, bias_parallel
                        )
                elif (
                    self.hidden_act == F.silu and self.config.gated_linear_unit
                ):
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        fp8_input_store=getattr(
                            self.config,
                            "activation_func_fp8_input_store",
                            False,
                        ),
                        cpu_offload_input=False,
                        clamp_value=self.config.activation_func_clamp_value,
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
                    return self.config.hidden_act(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.hidden_act(intermediate_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = (
                    intermediate_parallel * per_token_scale.unsqueeze(-1)
                )
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="down_proj")
        if (
            _ACCURACY_COMPATIBLE_KERNEL
            and self.config.tensor_model_parallel_size == 1
        ):
            output, output_bias = _accuracy_compatible_projection(
                self.down_proj, intermediate_parallel
            )
        else:
            output, output_bias = self.down_proj(intermediate_parallel)
        nvtx_range_pop(suffix="down_proj")

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias

    def backward_dw(self):
        self.down_proj.backward_dw()
        self.up_gate_proj.backward_dw()
