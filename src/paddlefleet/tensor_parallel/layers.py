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

from typing import TYPE_CHECKING, Callable

import paddle
import paddle.nn.functional as F

if TYPE_CHECKING:
    from paddlefleet.model_parallel_config import ModelParallelConfig


class ColumnParallelLinear(paddle.nn.Layer):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias
        gather_output:
            If true, call all-gather on output and make Y available to all GPUs,
            otherwise, every GPU will have its output which is Y_i = XA_i
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It
            returns the master weights used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimizations where bias can be fused with other
            elementwise operations.
        skip_weight_param_allocation:
            If True, weight parameter is not allocated and must be passed
            as a keyword argument `weight` during the forward pass. Note that this does not
            affect bias, which will be allocated if bias is True. Defaults to False.
        embedding_activation_buffer:
            This buffer holds the input activations of the final embedding
            linear layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        grad_output_buffer:
            This buffer holds the gradient outputs of the final embedding linear
            layer on the last pipeline stage when defer_embedding_wgrad_compute is enabled.
        is_expert:
            If True, the layer is treated as an MoE expert layer.
        config:
            ModelParallelConfig object
        disable_grad_reduce:
            If True, reduction of output gradients across tensor-parallel ranks
            will be disabled. Defaults to False. This feature is used by Lora Adapter in Nemo to
            delay and fuse reduction along with other gradients for performance optimization.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: list[paddle.Tensor] | None = None,
        grad_output_buffer: list[paddle.Tensor] | None = None,
        is_expert: bool = False,
        disable_grad_reduce: bool = False,
        tp_group: paddle.distributed.communication.group.Group | None = None,
    ):
        super().__init__()
        self.weight = None

        # TODO: Support ColumnParallelLinear?
        self.linear = paddle.nn.Linear(
            input_size,
            output_size,
        )

    def forward(self, input_, weight=None):
        if weight is None:
            return self.linear(input_), None
        else:
            return F.linear(input_, weight), None


class RowParallelLinear(paddle.nn.Layer):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its first dimension and X
    along its second dimension. A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]

    Args:
        input_size:
            first dimension of matrix A.
        output_size:
            second dimension of matrix A.
        bias:
            If true, add bias. Note that bias is not parallelized.
        input_is_parallel:
            If true, we assume that the input is already split across the GPUs
            and we do not split again.
        init_method:
            method to initialize weights. Note that bias is always set to zero.
        stride:
            For the strided linear layers.
        keep_master_weight_for_test:
            This was added for testing and should be set to False. It returns the master weights
            used for initialization.
        skip_bias_add:
            If True, do not add the bias term, instead return it to be added by the
            caller. This enables performance optimizations where bias can be fused with other
            elementwise operations.
        is_expert:
            If True, the layer is treated as an MoE expert layer
        config:
            ModelParallelConfig object

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_group: paddle.distributed.communication.group.Group | None = None,
    ):
        super().__init__()
        self.weight = None

        # TODO: Support ROWParallelLinear?
        self.linear = paddle.nn.Linear(
            input_size,
            output_size,
        )

    def forward(self, input_, weight=None):
        if weight is None:
            return self.linear(input_), None
        else:
            return F.linear(input_, weight), None
