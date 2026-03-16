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

"""
Manifold-Constrained Hyper-Connections (mHC) module.

Implements the mHC propagation from https://arxiv.org/abs/2502.20358:
    x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

This module uses a flat [s, b, n*C] tensor format (3D) for multi-stream
hidden states, unlike the existing mhc.py which uses [B, L, N, D] (4D).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import Tensor, nn

from paddlefleet.fusions.fused_bias_dropout import get_bias_dropout_add

from .triton_mhc import (
    TRITON_AVAILABLE,
    WidthConnectionLayerTriton,
    sinkhorn_knopp,
)

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig


class HyperConnectionModule(nn.Layer):
    """
    Unified mHC (Manifold-Constrained Hyper-Connections) module.

    Implements the complete mHC propagation:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)

    This module handles:
    1. Computing learnable mappings: H_pre, H_post, H_res (with Sinkhorn-Knopp projection)
    2. Aggregation: n-stream -> 1-stream (H_pre @ x)
    3. Expansion: 1-stream -> n-stream (H_post^T @ output)
    4. Residual merge: H_res @ x + expanded_output
    5. Block-level expand/contract for TransformerBlock boundaries

    Args:
        config: TransformerConfig with hyper-connection fields
        layer_number: Current layer index for initialization
    """

    def __init__(
        self, config: TransformerConfig, layer_number: int = 1, **kwargs
    ):
        super().__init__()
        self.config = config
        self.layer_number = layer_number
        self.n = config.mhc_num_residual_streams
        self.hidden_size = config.hidden_size
        self.sinkhorn_iterations = config.mhc_sinkhorn_iters
        self.use_triton = config.mhc_use_triton

        # Parameter merging optimization: combined weight matrix
        # Total output dim: pre(N) + post(N) + res(N*N)
        total_output_dim = self.n + self.n + self.n * self.n

        # Combined weight matrix: [n*C, n + n + n*n]
        self.combined_weights = self.create_parameter(
            shape=[
                self.n * self.hidden_size,
                total_output_dim,
            ],
            dtype="float32",
            default_initializer=nn.initializer.XavierNormal(),
        )

        # Combined scaling parameters: [pre_scale, post_scale, res_scale]
        self.scaling_factors = self.create_parameter(
            shape=[3],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.01),
        )

        # Combined bias parameters
        bias_dims = [self.n, self.n, self.n * self.n]
        total_bias_dim = sum(bias_dims)
        self.bias_terms = self.create_parameter(
            shape=[total_bias_dim],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )

        self.norm_eps = 1e-6

    def width_connection(
        self, x: Tensor, skip_sk_gradient: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Width connection for mHC - generates branch input and residual connections.

        This method implements the first half of mHC propagation:
        1. Compute learnable mappings: H_pre, H_post, H_res (with Sinkhorn-Knopp)
        2. Aggregate n-stream to 1-stream using H_pre
        3. Apply H_res to input for residual mixing

        Args:
            x: [s, b, n*C] - n-stream hidden states
            skip_sk_gradient: When True, detach Sinkhorn-Knopp gradients for
                numerical stability. Only applicable when using Triton backend.
                Default: True.

        Returns:
            branch_input: [s, b, C] - aggregated input for branch computation
            residuals: [s, b, n*C] - residual after H_res mixing
            h_post: [s, b, n] - expansion weights for depth connection
        """
        if self.use_triton and TRITON_AVAILABLE:
            return self._width_connection_triton(x, skip_sk_gradient)
        else:
            return self._width_connection_native(x)

    def _width_connection_native(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Native PaddlePaddle implementation of width connection.

        Args:
            x: [s, b, n*C] - n-stream hidden states

        Returns:
            branch_input: [s, b, C] - aggregated input for branch computation
            residuals: [s, b, n*C] - residual after H_res mixing
            h_post: [s, b, n] - expansion weights for depth connection
        """
        s, b, nC = x.shape
        n = self.n
        C = self.hidden_size
        x_dtype = x.dtype

        # Reshape [s, b, n*C] -> [s, b, n, C] for computation
        x_4d = x.reshape([s, b, n, C])

        # Normalize: RMSNorm over n*C dimension
        normed = self._rms_norm(x)  # [s, b, n*C]

        # Combined projection: [s, b, n*C] @ [n*C, n + n + n*n] -> [s, b, n + n + n*n]
        H_all = paddle.matmul(normed, self.combined_weights)  # [s, b, n^2 + 2n]

        # Apply scaling and bias
        pre_scale = self.scaling_factors[0]
        post_scale = self.scaling_factors[1]
        res_scale = self.scaling_factors[2]

        H_pre = H_all[:, :, :n] * pre_scale + self.bias_terms[:n]
        H_post = (
            H_all[:, :, n : 2 * n] * post_scale + self.bias_terms[n : 2 * n]
        )
        H_res = H_all[:, :, 2 * n :] * res_scale + self.bias_terms[2 * n :]

        # Apply activation functions
        H_pre = F.sigmoid(H_pre)
        H_post = 2 * F.sigmoid(H_post)

        # Reshape H_res to [s, b, n, n]
        H_res = H_res.reshape([s, b, n, n])

        with paddle.no_grad():
            # Subtract max for numerical stability
            H_res_max = H_res.max(axis=(-2, -1), keepdim=True)
            H_res_exp = paddle.exp(H_res - H_res_max)
            _, U, V = sinkhorn_knopp(
                H_res_exp.reshape([s * b, n, n]), self.sinkhorn_iterations
            )

        # Compute doubly stochastic matrix: U @ H_res_exp @ V
        res = paddle.matmul(
            paddle.matmul(U.detach(), H_res_exp.reshape([s * b, n, n])),
            V.detach(),
        )
        H_res_mat = res.reshape([s, b, n, n])

        # Compute residuals: H_res_mat @ x
        residuals = paddle.matmul(H_res_mat, x_4d)
        residuals = residuals.reshape([s, b, n * C])

        # Compute branch_input: H_pre @ x (aggregation)
        branch_input = paddle.matmul(H_pre.unsqueeze(dim=-2), x_4d).squeeze(-2)

        # Cast outputs back to input dtype to prevent dtype drift in AMP/bfloat16 scenarios
        if branch_input.dtype != x_dtype:
            branch_input = branch_input.cast(x_dtype)
        if residuals.dtype != x_dtype:
            residuals = residuals.cast(x_dtype)
        if H_post.dtype != x_dtype:
            H_post = H_post.cast(x_dtype)

        return branch_input, residuals, H_post

    def _width_connection_triton(
        self, x: Tensor, skip_sk_gradient: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Triton kernel implementation of width connection for maximum performance.

        Args:
            x: [s, b, n*C] - n-stream hidden states
            skip_sk_gradient: When True, detach Sinkhorn-Knopp gradients for
                numerical stability

        Returns:
            branch_input: [s, b, C] - aggregated input for branch computation
            residuals: [s, b, n*C] - residual after H_res mixing
            h_post: [s, b, n] - expansion weights for depth connection
        """
        s, b, nC = x.shape
        n = self.n
        C = self.hidden_size
        x_dtype = x.dtype

        # Reshape [s, b, n*C] -> [s, b, n, C] for Triton kernel
        x_4d = x.reshape([s, b, n, C])

        # Call Triton kernel
        branch_input_4d, residuals_4d, h_post = (
            WidthConnectionLayerTriton.apply(
                x_4d,
                self.combined_weights,
                self.scaling_factors,
                self.bias_terms,
                self.norm_eps,
                self.sinkhorn_iterations,
                n,
                C,
                skip_sk_gradient,
            )
        )

        # Reshape outputs back to [s, b, n*C] format
        branch_input = branch_input_4d.squeeze(-2)
        residuals = residuals_4d.reshape([s, b, n * C])

        if branch_input.dtype != x_dtype:
            branch_input = branch_input.cast(x_dtype)
        if residuals.dtype != x_dtype:
            residuals = residuals.cast(x_dtype)

        return branch_input, residuals, h_post

    def _rms_norm(self, x: Tensor) -> Tensor:
        """
        RMS normalization for native width connection implementation.

        Args:
            x: [s, b, n*C] - input tensor

        Returns:
            normalized: [s, b, n*C] - normalized tensor
        """
        nC = x.shape[-1]
        # Compute variance = mean(x^2)
        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        # RMSNorm: x / sqrt(variance + eps) = x * rsqrt(variance + eps)
        rstd = paddle.rsqrt(variance + self.norm_eps)
        return x * rstd

    def depth_connection(
        self,
        layer_output_with_bias: tuple[Tensor, Tensor | None],
        residuals: Tensor,
        h_post: Tensor,
        dropout_prob: float = 0.0,
        training: bool = False,
        fused: bool = False,
    ) -> Tensor:
        """
        Depth connection for mHC - expands branch output and combines with residuals.

        This method implements the second half of mHC propagation:
        1. Expand 1-stream branch output to n-stream using H_post
        2. Add optional bias (also expanded)
        3. Optionally apply dropout on the expanded output
        4. Combine with residuals: output = expanded + residuals

        Args:
            layer_output_with_bias: Tuple of (output, bias) or single Tensor where:
                - output: [s, b, C] - output from branch layer (attention/MLP)
                - bias: [C] or None - optional bias tensor to add
            residuals: [s, b, n*C] - residual from width_connection (already H_res mixed)
            h_post: [s, b, n] - expansion weights
            dropout_prob: Dropout probability. Default: 0.0 (no dropout)
            training: Whether in training mode. Default: False
            fused: Whether to use fused BDA implementation. Default: False

        Returns:
            output: [s, b, n*C] - final n-stream output
        """
        # Handle both tuple and single tensor input
        if isinstance(layer_output_with_bias, tuple):
            branch_output, bias = layer_output_with_bias
        else:
            branch_output, bias = layer_output_with_bias, None

        # Triton path
        if self.use_triton and TRITON_AVAILABLE:
            return self._depth_connection_triton(
                branch_output, residuals, h_post, bias, dropout_prob, training
            )

        # Native path
        return self._depth_connection_native(
            branch_output,
            residuals,
            h_post,
            bias,
            dropout_prob,
            training,
            fused,
        )

    def _depth_connection_native(
        self,
        branch_output: Tensor,
        residuals: Tensor,
        h_post: Tensor,
        bias: Tensor | None,
        dropout_prob: float,
        training: bool,
        fused: bool,
    ) -> Tensor:
        """Native PaddlePaddle implementation of depth connection with dropout."""
        s, b, C = branch_output.shape
        n = self.n
        x_dtype = branch_output.dtype

        # Ensure h_post matches branch_output dtype to prevent dtype drift
        if h_post.dtype != x_dtype:
            h_post = h_post.cast(x_dtype)

        # Pre-compute h_post_4d: [s, b, n, 1]
        h_post_4d = h_post.unsqueeze(-1)

        # Expand branch output: [s, b, n, 1] * [s, b, 1, C] -> [s, b, n, C] -> [s, b, n*C]
        expanded = (h_post_4d * branch_output.unsqueeze(2)).reshape(
            [s, b, n * C]
        )

        # Expand bias if present: [s, b, n, 1] * [1, 1, 1, C] -> [s, b, n, C] -> [s, b, n*C]
        bias_expanded = None
        if bias is not None:
            bias_expanded = (h_post_4d * bias.reshape([1, 1, 1, C])).reshape(
                [s, b, n * C]
            )

        # Apply bias-dropout-add
        bda_func = get_bias_dropout_add(training, fused)
        output = bda_func((expanded, bias_expanded), residuals, dropout_prob)

        return output

    def _depth_connection_triton(
        self,
        branch_output: Tensor,
        residuals: Tensor,
        h_post: Tensor,
        bias: Tensor | None,
        dropout_prob: float,
        training: bool,
    ) -> Tensor:
        """Triton kernel implementation of depth connection with dropout."""
        s, b, C = branch_output.shape
        n = self.n
        x_dtype = branch_output.dtype

        # Ensure h_post matches branch_output dtype to prevent dtype drift
        if h_post.dtype != x_dtype:
            h_post = h_post.cast(x_dtype)

        # Pre-compute h_post_4d: [s, b, n, 1]
        h_post_4d = h_post.unsqueeze(-1)

        # Expand branch output: [s, b, n, 1] * [s, b, 1, C] -> [s, b, n, C] -> [s, b, n*C]
        expanded = (h_post_4d * branch_output.unsqueeze(2)).reshape(
            [s, b, n * C]
        )

        # Expand bias if present: [s, b, n, 1] * [1, 1, 1, C] -> [s, b, n, C] -> [s, b, n*C]
        bias_expanded = None
        if bias is not None:
            bias_expanded = (h_post_4d * bias.reshape([1, 1, 1, C])).reshape(
                [s, b, n * C]
            )

        # Apply bias-dropout-add with correct semantics:
        # dropout(expanded + bias) + residuals (not dropout(expanded + bias + residuals))
        bda_func = get_bias_dropout_add(training, fused=False)
        output = bda_func((expanded, bias_expanded), residuals, dropout_prob)

        if output.dtype != x_dtype:
            output = output.cast(x_dtype)

        return output

    # ==================== Block-level utilities ====================

    @staticmethod
    def expand_stream(x: Tensor, n: int) -> Tensor:
        """
        Expand 1-stream to n-stream at TransformerBlock entry.

        Simple replication strategy: each stream initialized as a copy of input.

        Args:
            x: [s, b, C] - single stream hidden states
            n: Number of residual streams

        Returns:
            expanded: [s, b, n*C] - n-stream hidden states
        """
        s, b, C = x.shape
        # Replicate input to n streams
        expanded = x.unsqueeze(2).expand([s, b, n, C])
        return expanded.reshape([s, b, n * C])

    @staticmethod
    def reduce_stream(x: Tensor, n: int) -> Tensor:
        """
        Contract n-stream to 1-stream at TransformerBlock exit.

        Simple averaging strategy: average all streams.

        Args:
            x: [s, b, n*C] - n-stream hidden states
            n: Number of residual streams

        Returns:
            contracted: [s, b, C] - single stream hidden states
        """
        s, b, nC = x.shape
        C = nC // n
        # Average all streams
        x_streams = x.reshape([s, b, n, C])
        contracted = x_streams.mean(axis=2)
        return contracted


class MHCExpandLayer(nn.Layer):
    """Pipeline-parallel compatible layer for expanding 1-stream to n-stream.

    This thin wrapper can be inserted into the pipeline layer desc list to
    perform stream expansion at the block entry in PP mode.
    """

    def __init__(self, config: TransformerConfig, **kwargs):
        super().__init__()
        self.n = config.mhc_num_residual_streams

    def forward(self, dict_args):
        hidden_states = dict_args["hidden_states"]
        dict_args["hidden_states"] = HyperConnectionModule.expand_stream(
            hidden_states, self.n
        )
        return dict_args


class MHCContractLayer(nn.Layer):
    """Pipeline-parallel compatible layer for contracting n-stream to 1-stream.

    This thin wrapper can be inserted into the pipeline layer desc list to
    perform stream contraction at the block exit in PP mode.
    """

    def __init__(self, config: TransformerConfig, **kwargs):
        super().__init__()
        self.n = config.mhc_num_residual_streams

    def forward(self, dict_args):
        hidden_states = dict_args["hidden_states"]
        dict_args["hidden_states"] = HyperConnectionModule.reduce_stream(
            hidden_states, self.n
        )
        return dict_args
