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

import logging

import paddle

try:
    from paddle.incubate.nn.functional.fused_rms_norm_ext import (
        fused_rms_norm_ext,
    )
except ImportError:
    logging.warn("Fail to import fused_rms_norm_ext!")
    fused_rms_norm_ext = None

try:
    from paddle.distributed.fleet.utils.sequence_parallel_utils import (
        mark_as_sequence_parallel_parameter,
    )
except ImportError:
    logging.warn("Fail to import mark_as_sequence_parallel_parameter!")

    def mark_as_sequence_parallel_parameter(parameter):
        return parameter


from paddlefleet.jit import jit_fuser
from paddlefleet.transformer import TransformerConfig


class RMSNorm(paddle.nn.Layer):
    def __init__(
        self,
        config: TransformerConfig,
        normalized_shape=None,
        norm_eps=None,
        input_is_parallel=False,
        **kwargs,
    ):
        super().__init__()
        self.normalized_shape = (
            config.hidden_size if normalized_shape is None else normalized_shape
        )
        self.variance_epsilon = (
            config.rms_norm_eps if norm_eps is None else norm_eps
        )
        self.weight = paddle.create_parameter(
            shape=[self.normalized_shape],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.config = config

        if input_is_parallel:
            self.enable_sequence_parallel()

    def forward(self, hidden_states):
        if self.config.fuse_rms_norm:
            assert fused_rms_norm_ext is not None, (
                "Enable fuse rms norm but paddle version is incorrect."
            )
            return fused_rms_norm_ext(
                hidden_states, self.weight, self.variance_epsilon
            )[0].astype(self.weight.dtype)

        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                variance = (
                    hidden_states.astype("float32")
                    .pow(2)
                    .mean(-1, keepdim=True)
                )
                hidden_states = (
                    paddle.rsqrt(variance + self.variance_epsilon)
                    * hidden_states
                )
        else:
            variance = (
                hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
            )
            hidden_states = (
                paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
            )

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight

    def enable_sequence_parallel(self):
        mark_as_sequence_parallel_parameter(self.weight)


class WrappedPaddleNorm:
    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        input_is_parallel: bool = False,
    ):
        if config.normalization == "RMSNorm":
            norm_cls = RMSNorm
        else:
            raise Exception("Only RMSNorm for now.")

        input_is_parallel = config.sequence_parallel
        return norm_cls(
            config=config,
            normalized_shape=hidden_size,
            norm_eps=eps,
            input_is_parallel=input_is_parallel,
        )


class L2Norm(paddle.nn.Layer):
    """
    Applies L2 normalization to the input tensor along the last dimension.

    This layer normalizes the input tensor such that the mean of the squared values
    along the last dimension is 1 (within a small epsilon for numerical stability).

    Args:
        hidden_size (int): Expected input shape for normalization (not used internally).
        eps (float, optional): A small value added to the denominator for numerical stability.
            Default: 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    @jit_fuser
    def _norm(self, x):
        """
        Performs the actual L2 normalization.

        Args:
            x (paddle.Tensor): The input tensor to normalize.

        Returns:
            paddle.Tensor: The L2-normalized tensor.
        """
        x_float = x.float()
        return (
            x_float
            * paddle.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        ).astype(x.dtype)

    def forward(self, x):
        """
        Forward pass of the L2Norm module.

        Args:
            x (paddle.Tensor): Input tensor.

        Returns:
            paddle.Tensor: L2-normalized tensor with the same dtype as input.
        """
        return self._norm(x)
