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

# Ported from NVIDIA Megatron-LM megatron/core/ssm/gated_delta_net.py
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    build_spec_layer,
)

from paddlefleet.jit import jit_fuser
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.layers import ColumnParallelLinear
from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.utils import (
    get_pg_rank,
    get_pg_size,
    nvtx_range_pop,
    nvtx_range_push,
)

from .paddle_norm import get_norm_extra_args

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    HAVE_FLA = True
except ImportError:
    chunk_gated_delta_rule = None
    HAVE_FLA = False

logger = logging.getLogger(__name__)


def _l2norm(x):
    """Apply L2 normalization along the last dimension.

    Equivalent to fla.modules.l2norm.l2norm for paddle tensors.
    """
    x_float = x.astype(paddle.float32)
    inv_norm = paddle.rsqrt(x_float.pow(2).sum(-1, keepdim=True) + 1e-6)
    return (x_float * inv_norm).astype(x.dtype)


@dataclass
class GatedDeltaNetSublayersSpec:
    """Contains the layer specs for the input linear, output norm, and output linear layers."""

    in_proj: LayerSpec | type = IdentityOp
    out_norm: LayerSpec | type = IdentityOp
    out_proj: LayerSpec | type = IdentityOp


def _accuracy_gdn_in_proj_order_enabled(config) -> bool:
    """Return whether the opt-in Megatron GDN projection order is active."""
    return bool(getattr(config, "use_accuracy_compatible", False)) and (
        os.environ.get("PADDLEFLEET_ACCURACY_GDN_IN_PROJ_ORDER", "0") == "1"
    )


def _accuracy_gdn_causal_conv1d_enabled(config) -> bool:
    """Return whether the opt-in Megatron causal Conv1D path is active."""
    enabled = bool(getattr(config, "use_accuracy_compatible", False)) and (
        os.environ.get("PADDLEFLEET_ACCURACY_GDN_CAUSAL_CONV1D", "0") == "1"
    )
    if enabled and not _accuracy_gdn_in_proj_order_enabled(config):
        raise ValueError(
            "PADDLEFLEET_ACCURACY_GDN_CAUSAL_CONV1D requires "
            "PADDLEFLEET_ACCURACY_GDN_IN_PROJ_ORDER=1"
        )
    return enabled


def _accuracy_gdn_conv_fp32_accum_enabled(config) -> bool:
    """Return whether the opt-in FP32-accumulator GDN Conv1D path is active."""
    enabled = bool(getattr(config, "use_accuracy_compatible", False)) and (
        os.environ.get("PADDLEFLEET_ACCURACY_GDN_CONV_FP32_ACCUM", "0") == "1"
    )
    if enabled and not _accuracy_gdn_causal_conv1d_enabled(config):
        raise ValueError(
            "PADDLEFLEET_ACCURACY_GDN_CONV_FP32_ACCUM requires "
            "PADDLEFLEET_ACCURACY_GDN_CAUSAL_CONV1D=1"
        )
    return enabled


def _accuracy_gdn_qk_l2_enabled(config) -> bool:
    """Return whether the opt-in Megatron-compatible Q/K L2 path is active."""
    enabled = bool(getattr(config, "use_accuracy_compatible", False)) and (
        os.environ.get("PADDLEFLEET_ACCURACY_GDN_QK_L2", "0") == "1"
    )
    if enabled and not _accuracy_gdn_causal_conv1d_enabled(config):
        raise ValueError(
            "PADDLEFLEET_ACCURACY_GDN_QK_L2 requires "
            "PADDLEFLEET_ACCURACY_GDN_CAUSAL_CONV1D=1"
        )
    return enabled


def _accuracy_gdn_recurrence_enabled(config) -> bool:
    """Return whether the opt-in Megatron chunk recurrence scan is active."""
    enabled = bool(getattr(config, "use_accuracy_compatible", False)) and (
        os.environ.get("PADDLEFLEET_ACCURACY_GDN_RECURRENCE", "0") == "1"
    )
    if enabled and not _accuracy_gdn_qk_l2_enabled(config):
        raise ValueError(
            "PADDLEFLEET_ACCURACY_GDN_RECURRENCE requires "
            "PADDLEFLEET_ACCURACY_GDN_QK_L2=1"
        )
    return enabled


def _accuracy_compatible_sklansky_scan(x: paddle.Tensor):
    """Inclusive last-axis Sklansky scan matching Torch CUDA cumsum."""
    width = x.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError(
            "accuracy-compatible GDN recurrence requires a positive power-of-two chunk"
        )
    output = x
    step = 2
    while step <= width:
        half = step // 2
        parts = []
        for start in range(0, width, step):
            lower = output[..., start : start + half]
            upper = output[..., start + half : start + step]
            parts.extend([lower, upper + lower[..., -1:]])
        output = paddle.concat(parts, axis=-1)
        step *= 2
    return output


def _accuracy_compatible_l2norm(x: paddle.Tensor, eps: float = 1e-6):
    """Match FLA's BF16 L2 output with an explicit reduction tree.

    FLA's Triton kernel reduces a 128-wide FP32 row and applies reciprocal
    square root before storing BF16.  For the frozen H800 profile, reverse
    serial accumulation plus ``1 / sqrt`` is the only tested Paddle-native
    tree whose full BF16 Q/K output is raw exact.  This candidate intentionally
    makes no backward exactness claim yet.
    """
    if x.shape[-1] <= 0:
        raise ValueError("accuracy-compatible GDN L2 requires a non-empty row")
    x_float = x.astype(paddle.float32)
    squares = x_float * x_float
    sum_squares = squares[..., -1]
    for index in range(x.shape[-1] - 2, -1, -1):
        sum_squares = squares[..., index] + sum_squares
    inv_norm = 1.0 / paddle.sqrt(sum_squares.unsqueeze(-1) + eps)
    return (x_float * inv_norm).astype(x.dtype)


def _reorder_gdn_in_proj_by_key_head(
    output: paddle.Tensor,
    *,
    num_key_heads: int,
    key_head_dim: int,
    num_value_heads: int,
    value_head_dim: int,
) -> paddle.Tensor:
    """Reorder ``[q,k,v,z,b,a]`` sections into Megatron key-head groups.

    The Qwen3.5 mcore bridge reshapes each of q/k/v/z/b/a to
    ``[num_key_heads, -1]`` and concatenates within every key-head group before
    flattening. Paddle's official checkpoint remains in section-major order;
    this read-only activation permutation reproduces the bridge result without
    changing parameter names, shapes, or checkpoint bytes.
    """
    if num_key_heads <= 0 or num_value_heads % num_key_heads != 0:
        raise ValueError(
            "GDN Megatron projection order requires positive key heads and "
            "num_value_heads divisible by num_key_heads"
        )
    qk_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    section_sizes = [
        qk_dim,
        qk_dim,
        value_dim,
        value_dim,
        num_value_heads,
        num_value_heads,
    ]
    expected = sum(section_sizes)
    if output.shape[-1] != expected:
        raise ValueError(
            "GDN Megatron projection order received an unexpected output size: "
            f"expected {expected}, got {output.shape[-1]}"
        )
    grouped = [
        section.reshape([*output.shape[:-1], num_key_heads, -1])
        for section in paddle.split(output, section_sizes, axis=-1)
    ]
    return paddle.concat(grouped, axis=-1).reshape(output.shape)


def _reorder_gdn_conv1d_by_key_head(
    weight: paddle.Tensor,
    *,
    num_key_heads: int,
    key_head_dim: int,
    num_value_heads: int,
    value_head_dim: int,
) -> paddle.Tensor:
    """Reorder official q/k/v Conv1D channels into mcore key-head groups."""
    if num_key_heads <= 0 or num_value_heads % num_key_heads != 0:
        raise ValueError(
            "GDN Megatron Conv1D order requires positive key heads and "
            "num_value_heads divisible by num_key_heads"
        )
    qk_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    expected = qk_dim * 2 + value_dim
    if weight.shape[0] != expected or weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError(
            "GDN Megatron Conv1D order received an unexpected weight shape: "
            f"expected [{expected}, 1, kernel], got {list(weight.shape)}"
        )
    grouped = [
        section.reshape([num_key_heads, -1, 1, weight.shape[-1]])
        for section in paddle.split(
            weight, [qk_dim, qk_dim, value_dim], axis=0
        )
    ]
    return paddle.concat(grouped, axis=1).reshape(weight.shape)


class AccuracyCompatibleGDNConv1D(nn.Conv1D):
    """Depthwise Conv1D matching mcore's grouped weight and BF16 GEMM tree."""

    def __init__(
        self,
        *args,
        config,
        num_key_heads: int,
        key_head_dim: int,
        num_value_heads: int,
        value_head_dim: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.accuracy_compatible_gdn_causal_conv1d = (
            _accuracy_gdn_causal_conv1d_enabled(config)
        )
        self.accuracy_compatible_gdn_conv_fp32_accum = (
            _accuracy_gdn_conv_fp32_accum_enabled(config)
        )
        self._gdn_order = {
            "num_key_heads": num_key_heads,
            "key_head_dim": key_head_dim,
            "num_value_heads": num_value_heads,
            "value_head_dim": value_head_dim,
        }
        if self.accuracy_compatible_gdn_causal_conv1d:
            if self._groups != self._in_channels or self._groups != self._out_channels:
                raise ValueError("accuracy-compatible GDN Conv1D requires depthwise groups")
            if self._stride != [1] or self._dilation != [1] or self.bias is not None:
                raise ValueError(
                    "accuracy-compatible GDN Conv1D requires stride=1, dilation=1, and no bias"
                )
            if self._padding != self._kernel_size[0] - 1 or self._data_format != "NCL":
                raise ValueError(
                    "accuracy-compatible GDN Conv1D requires symmetric kernel-1 NCL padding"
                )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if not self.accuracy_compatible_gdn_causal_conv1d:
            return super().forward(x)
        weight = _reorder_gdn_conv1d_by_key_head(self.weight, **self._gdn_order)
        padding = self._padding
        kernel = self._kernel_size[0]
        padded = F.pad(x, [padding, padding])
        output_length = padded.shape[-1] - kernel + 1
        windows = paddle.stack(
            [padded[:, :, tap : tap + output_length] for tap in range(kernel)],
            axis=-1,
        )
        batch, channels = x.shape[:2]
        bmm_weight = (
            weight[:, 0, :]
            .unsqueeze(0)
            .expand([batch, channels, kernel])
            .reshape([batch * channels, kernel, 1])
        )
        windows = windows.reshape([batch * channels, output_length, kernel])
        bmm_weight = bmm_weight.reshape([batch * channels, kernel, 1])
        if self.accuracy_compatible_gdn_conv_fp32_accum:
            return paddle.bmm(
                windows.astype(paddle.float32),
                bmm_weight.astype(paddle.float32),
            ).astype(x.dtype).reshape([batch, channels, output_length])
        return paddle.bmm(windows, bmm_weight).reshape(
            [batch, channels, output_length]
        )



class AccuracyCompatibleGDNInputProjection(ColumnParallelLinear):
    """Column-parallel projection with an explicit Qwen3.5 mcore row order.

    The behavior is default-off and restricted to the independently validated
    TP1 accuracy profile. The underlying parameter/state-dict layout remains
    unchanged, so official checkpoint loading and inverse export stay lossless.
    """

    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.accuracy_compatible_gdn_in_proj_order = (
            _accuracy_gdn_in_proj_order_enabled(config)
        )
        if self.accuracy_compatible_gdn_in_proj_order and self.world_size != 1:
            raise ValueError(
                "PADDLEFLEET_ACCURACY_GDN_IN_PROJ_ORDER is validated only for TP1"
            )

    def forward(self, *args, **kwargs):
        output, output_bias = super().forward(*args, **kwargs)
        if not self.accuracy_compatible_gdn_in_proj_order:
            return output, output_bias
        reorder_kwargs = {
            "num_key_heads": self.config.linear_num_key_heads,
            "key_head_dim": self.config.linear_key_head_dim,
            "num_value_heads": self.config.linear_num_value_heads,
            "value_head_dim": self.config.linear_value_head_dim,
        }
        output = _reorder_gdn_in_proj_by_key_head(output, **reorder_kwargs)
        if output_bias is not None:
            output_bias = _reorder_gdn_in_proj_by_key_head(
                output_bias, **reorder_kwargs
            )
        return output, output_bias


class GatedDeltaNet(FleetLayer):
    """Gated Delta Net (GDN) layer class.

    GDN layer takes input with size [b, s, h] (or [s, b, h] when sequence_parallel is enabled)
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: GatedDeltaNetSublayersSpec,
        layer_number: int | None = None,
        bias: bool = False,
        conv_bias: bool = False,
        conv_init: float | None = None,
        use_qk_l2norm: bool = True,
        A_init_range: tuple[float, float] = (1, 16),
        pg_collection: ProcessGroupCollection = None,
        conv_kernel_dim: int = 4,
        key_head_dim: int = 128,
        value_head_dim: int = 128,
        num_key_heads: int = 16,
        num_value_heads: int = 32,
    ):
        """
        Args:
            config: The transformer config of the model.
            sublayers_spec: Contains the layer specs for the input and output linear layers.
            layer_number: The layer number of this GDN layer.
            bias: Whether to use bias in the linear layers.
            conv_bias: Whether to use bias in the causal convolution.
            conv_init: The initialization range for the causal convolution weights.
            use_qk_l2norm: Whether to use L2 normalization on query and key.
            A_init_range: The initialization range for the A parameter.
            pg_collection: The required process groups for tensor model parallel.
            conv_kernel_dim: Kernel size for the causal convolution.
            key_head_dim: Dimension of each query/key head.
            value_head_dim: Dimension of each value/gate head.
            num_key_heads: Number of query/key heads.
            num_value_heads: Number of value/gate heads.
        """
        super().__init__(config=config)

        # Attributes from arguments
        self.layer_number = layer_number
        self.bias = bias
        self.conv_bias = conv_bias
        self.conv_init = conv_init
        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        self.A_init_range = A_init_range
        self.use_qk_l2norm = use_qk_l2norm
        self.accuracy_compatible_gdn_in_proj_order = _accuracy_gdn_in_proj_order_enabled(
            config
        )
        self.accuracy_compatible_gdn_qk_l2 = _accuracy_gdn_qk_l2_enabled(
            config
        )
        self.accuracy_compatible_gdn_recurrence = (
            _accuracy_gdn_recurrence_enabled(config)
        )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp"]
            )
        assert pg_collection is not None, (
            "pg_collection must be provided for GatedDeltaNet"
        )
        self.pg_collection = pg_collection
        self.tp_size = get_pg_size(self.pg_collection.tp)
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.hidden_size = config.hidden_size
        self.act_fn = config.hidden_act
        self.activation = getattr(self.act_fn, "__name__", "silu")
        self.conv_kernel_dim = conv_kernel_dim
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads

        # Input projection (hidden_states -> q, k, v, gate, beta, alpha)
        self.in_proj_dim = (
            self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
        )

        self.in_proj = build_spec_layer(
            sublayers_spec.in_proj,
            self.hidden_size,
            self.in_proj_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        # Conv1D for QKV
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size

        # weight shape: [conv_dim, 1, d_conv], bias shape: [conv_dim]
        self.conv1d = AccuracyCompatibleGDNConv1D(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            bias_attr=conv_bias,
            data_format="NCL",
            config=self.config,
            num_key_heads=self.num_key_heads,
            key_head_dim=self.key_head_dim,
            num_value_heads=self.num_value_heads,
            value_head_dim=self.value_head_dim,
        )
        self.conv1d.weight.is_distributed = True if self.tp_size > 1 else False
        if conv_bias and self.conv1d.bias is not None:
            self.conv1d.bias.is_distributed = (
                True if self.tp_size > 1 else False
            )

        # Time step projection (discretization)
        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size

        # dt_bias parameter — fp32 for numerical stability in softplus
        self.dt_bias = self.create_parameter(
            shape=[self.num_v_heads_local_tp],
            dtype="float32",
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.dt_bias.is_distributed = True if self.tp_size > 1 else False

        # A_log parameter — fp32 to avoid exp() overflow in bf16
        self.A_log = self.create_parameter(
            shape=[self.num_v_heads_local_tp],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.A_log.is_distributed = True if self.tp_size > 1 else False

        # Output layernorm before projection (per-head norm)
        # out_norm weight is replicated (not TP-sharded), but each TP rank only
        # processes its local value heads, so the gradient is a partial sum.
        # Mark the parameter so that register_sequence_parallel_allreduce_hooks
        # will all-reduce its gradient across the TP group.
        input_is_parallel = True if self.tp_size > 1 else False
        extra_args = get_norm_extra_args(
            sublayers_spec.out_norm,
            self.config,
            self.value_head_dim,
            self.config.rms_norm_eps,
            input_is_parallel,
        )
        self.out_norm = build_spec_layer(
            sublayers_spec.out_norm,
            **extra_args,
        )

        self.out_proj = build_spec_layer(
            sublayers_spec.out_proj,
            self.v_dim,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True if self.tp_size > 1 else False,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            # conv1d.weight
            if self.conv_init is not None:
                nn.initializer.Uniform(
                    low=-self.conv_init, high=self.conv_init
                )(self.conv1d.weight)

            # dt_bias: initialize to ones
            nn.initializer.Constant(1.0)(self.dt_bias)

            # A_log: initialize to log(uniform(A_init_range))
            A = paddle.empty([self.num_v_heads_local_tp], dtype="float32")
            nn.initializer.Uniform(
                low=self.A_init_range[0], high=self.A_init_range[1]
            )(A)
            paddle.assign(paddle.log(A), self.A_log)

    def _build_padding_mask(
        self,
        attention_mask: paddle.Tensor | None,
        attn_mask_startend_row_indices: paddle.Tensor | None,
        batch: int,
        seq_len: int,
    ) -> paddle.Tensor | None:
        """Derive a padding mask (1.0=valid, 0.0=padding) for GDN."""
        is_sp = self.config.sequence_parallel and self.sp_size > 1

        if attention_mask is not None and attention_mask.ndim in (2, 4):
            if attention_mask.ndim == 2:
                valid_tokens = attention_mask
                full_seq = attention_mask.shape[-1]
            elif (
                attention_mask.shape[1] == 1
                and attention_mask.shape[-2] == attention_mask.shape[-1]
            ):
                # PaddleFormers dense masks use 1.0 for an allowed QK edge and
                # 0.0 for a masked edge. A valid causal query always retains its
                # diagonal edge, while padding rows have a zero diagonal.
                valid_tokens = paddle.diagonal(
                    attention_mask[:, 0], axis1=-2, axis2=-1
                )
                full_seq = attention_mask.shape[-1]
            else:
                valid_tokens = None
                full_seq = attention_mask.shape[-1]

            if valid_tokens is not None and is_sp:
                if full_seq != seq_len:
                    # Shape mismatch under SP – fall through to startend indices.
                    pass
                else:
                    # valid_tokens is [b, full_s], slice to local chunk.
                    seq_len_local = seq_len // self.sp_size
                    tp_rank = get_pg_rank(self.pg_collection.tp)
                    offset = tp_rank * seq_len_local
                    local_mask = valid_tokens[:, offset : offset + seq_len_local]
                    if local_mask.astype("bool").all():
                        return None
                    return local_mask.astype(paddle.float32).T.unsqueeze(-1)
            elif valid_tokens is not None and full_seq == seq_len:
                mask = valid_tokens.unsqueeze(-1).astype(paddle.float32)
                # paddle.diagonal returns a strided view; materialize a bool
                # reduction because GPU all() on that float view is unreliable.
                if mask.astype("bool").all():
                    return None
                return mask
            # The dense/padding mask shape does not match the current sequence
            # (for example a stale MTP-lookahead mask). Fall through to try
            # attn_mask_startend_row_indices, then fail closed below.

        if attn_mask_startend_row_indices is not None:
            indices = attn_mask_startend_row_indices[:, 0, :, 0]
            full_seq = indices.shape[-1]

            if is_sp:
                if full_seq != seq_len:
                    # Shape mismatch under SP – cannot derive valid mask.
                    pass
                else:
                    seq_len_local = seq_len // self.sp_size
                    tp_rank = get_pg_rank(self.pg_collection.tp)
                    offset = tp_rank * seq_len_local
                    local_indices = indices[:, offset : offset + seq_len_local]
                    seq_positions = paddle.arange(
                        offset,
                        offset + seq_len_local,
                        dtype=local_indices.dtype,
                    )
                    valid = (local_indices > seq_positions.unsqueeze(0)).astype(
                        paddle.float32
                    )
                    if valid.all():
                        return None
                    return valid.T.unsqueeze(-1)
            else:
                seq_positions = paddle.arange(full_seq, dtype=indices.dtype)
                valid = (indices > seq_positions.unsqueeze(0)).astype(
                    paddle.float32
                )
                if valid.all():
                    return None
                return valid.unsqueeze(-1)

        if (
            attention_mask is not None
            or attn_mask_startend_row_indices is not None
        ):
            raise ValueError(
                f"GatedDeltaNet._build_padding_mask: could not derive a valid "
                f"padding mask from the provided inputs "
                f"(attention_mask.shape={list(attention_mask.shape) if attention_mask is not None else None}, "
                f"attn_mask_startend_row_indices.shape="
                f"{list(attn_mask_startend_row_indices.shape) if attn_mask_startend_row_indices is not None else None}, "
                f"seq_len={seq_len})."
            )
        return None

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        key_value_states: paddle.Tensor | None = None,
        attention_bias: paddle.Tensor | None = None,
        packed_seq_params=None,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """
        Perform a forward pass through the GDN module.

        Args:
            hidden_states: Hidden states [b, s, h] or [s, b, h] with sequence_parallel.
            attention_mask: Attention mask.
            key_value_states: Key/value states (for cross attention, not supported).
            attention_bias: Attention bias.
            packed_seq_params: Parameters used for THD format (not supported).

        Returns:
            Tuple of (output, output_bias).
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "GDN does not support packed sequence for now."
            )

        hidden_states = hidden_states.contiguous()
        # Determine sequence layout
        if self.config.sequence_parallel and self.sp_size > 1:
            # Input is [s, b, h] with sequence parallel
            seq_len_local, batch, _ = hidden_states.shape
            seq_len = seq_len_local * self.sp_size
        else:
            # Input is [b, s, h]
            batch, seq_len, _ = hidden_states.shape

        attn_mask_startend_row_indices = kwargs.get(
            "attn_mask_startend_row_indices", None
        )
        padding_mask = self._build_padding_mask(
            attention_mask, attn_mask_startend_row_indices, batch, seq_len
        )
        if padding_mask is not None:
            hidden_states = hidden_states * padding_mask.astype(
                hidden_states.dtype
            )

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        # Ensure [b, s, x] format for the rest of computation
        if self.config.sequence_parallel:
            # [s, b, x] -> [b, s, x]
            qkvzba = qkvzba.transpose([1, 0, 2])

        # Split, reorder, and reshape the tensor into q, k, v, gate, beta, alpha.
        # The accuracy-compatible projection has Megatron's per-key-head layout:
        # [q(key_dim), k(key_dim), v(value_dim/key_head), z(value_dim/key_head),
        #  b(value_heads/key_head), a(value_heads/key_head)] per key head.
        if self.accuracy_compatible_gdn_in_proj_order:
            local_key_heads = self.num_key_heads // self.tp_size
            value_heads_per_key_head = self.num_value_heads // self.num_key_heads
            value_dim_per_key_head = value_heads_per_key_head * self.value_head_dim
            qkvzba = qkvzba.reshape([batch, seq_len, local_key_heads, -1])
            qkv, gate, beta, alpha = paddle.split(
                qkvzba,
                [
                    2 * self.key_head_dim + value_dim_per_key_head,
                    value_dim_per_key_head,
                    value_heads_per_key_head,
                    value_heads_per_key_head,
                ],
                axis=-1,
            )
            gate = gate.reshape([batch, seq_len, self.num_value_heads // self.tp_size, self.value_head_dim])
            beta = beta.reshape([batch, seq_len, -1])
            alpha = alpha.reshape([batch, seq_len, -1])
            qkv = qkv.reshape([batch, seq_len, -1])
        else:
            qkv, gate, beta, alpha = paddle.split(
                qkvzba,
                [
                    (self.qk_dim * 2 + self.v_dim) // self.tp_size,
                    self.v_dim // self.tp_size,
                    self.num_value_heads // self.tp_size,
                    self.num_value_heads // self.tp_size,
                ],
                axis=-1,
            )
            gate = gate.reshape([batch, seq_len, -1, self.value_head_dim])
            beta = beta.reshape([batch, seq_len, -1])
            alpha = alpha.reshape([batch, seq_len, -1])

        # Convolution on qkv
        qkv = qkv.transpose([0, 2, 1]).contiguous()  # b, s, d -> b, d, s
        nvtx_range_push(suffix="conv1d")
        # Always use Conv1D + activation path (causal_conv1d not available for Paddle)
        qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
        nvtx_range_pop(suffix="conv1d")

        # Split qkv into query/key and value using the same per-key-head layout.
        qkv = qkv.transpose([0, 2, 1])  # b, d, s -> b, s, d
        if self.accuracy_compatible_gdn_in_proj_order:
            local_key_heads = self.num_key_heads // self.tp_size
            value_heads_per_key_head = self.num_value_heads // self.num_key_heads
            qkv = qkv.reshape([batch, seq_len, local_key_heads, -1])
            query, key, value = paddle.split(
                qkv,
                [
                    self.key_head_dim,
                    self.key_head_dim,
                    value_heads_per_key_head * self.value_head_dim,
                ],
                axis=-1,
            )
            query = query.reshape([batch, seq_len, -1, self.key_head_dim])
            key = key.reshape([batch, seq_len, -1, self.key_head_dim])
            value = value.reshape([batch, seq_len, -1, self.value_head_dim])
        else:
            query_key, value = paddle.split(
                qkv,
                [
                    self.qk_dim * 2 // self.tp_size,
                    self.v_dim // self.tp_size,
                ],
                axis=-1,
            )
            query_key = query_key.reshape(
                [batch, seq_len, -1, self.key_head_dim]
            )
            value = value.reshape([batch, seq_len, -1, self.value_head_dim])

        recurrence_observer = getattr(self, "_repro_recurrence_observer", None)
        query_pre_norm = query if recurrence_observer is not None else None
        key_pre_norm = key if recurrence_observer is not None else None

        # FLA normalizes Q/K rows before the recurrent rule.
        if self.accuracy_compatible_gdn_in_proj_order:
            if self.use_qk_l2norm:
                if self.accuracy_compatible_gdn_qk_l2:
                    query = _accuracy_compatible_l2norm(query.contiguous())
                    key = _accuracy_compatible_l2norm(key.contiguous())
                else:
                    query = _l2norm(query.contiguous())
                    key = _l2norm(key.contiguous())
        else:
            # The legacy section-major path normalizes concatenated Q/K rows
            # before splitting their head axis.
            if self.use_qk_l2norm:
                if self.accuracy_compatible_gdn_qk_l2:
                    query_key = _accuracy_compatible_l2norm(
                        query_key.contiguous()
                    )
                else:
                    query_key = _l2norm(query_key.contiguous())
            local_key_heads = self.num_key_heads // self.tp_size
            query, key = paddle.split(
                query_key, [local_key_heads, local_key_heads], axis=2
            )

        # GQA repeat if num_value_heads > num_key_heads
        if self.num_value_heads // self.num_key_heads > 1:
            query = paddle.repeat_interleave(
                query, self.num_value_heads // self.num_key_heads, axis=2
            )
            key = paddle.repeat_interleave(
                key, self.num_value_heads // self.num_key_heads, axis=2
            )

        # Make contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        # Calculate g and beta
        nvtx_range_push(suffix="g_and_beta")
        g = -self.A_log.astype(paddle.float32).exp() * F.softplus(
            alpha.astype(paddle.float32) + self.dt_bias.astype(paddle.float32)
        )
        beta = beta.sigmoid()
        nvtx_range_pop(suffix="g_and_beta")

        nvtx_range_push(suffix="gated_delta_rule")
        if (not HAVE_FLA) or self.config.deterministic_mode:
            core_attn_out, last_recurrent_state = paddle_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                use_accuracy_compatible_cumsum=(
                    self.accuracy_compatible_gdn_recurrence
                ),
            )
        else:
            raise NotImplementedError("FLA not supported yet.")
            # core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            #     query,
            #     key,
            #     value,
            #     g=g,
            #     beta=beta,
            #     initial_state=None,
            #     output_final_state=False,
            #     use_qk_l2norm_in_kernel=False,
            # )
        if recurrence_observer is not None:
            recurrence_observer(
                query_pre_norm=query_pre_norm,
                key_pre_norm=key_pre_norm,
                query=query,
                key=key,
                value=value,
                alpha=alpha,
                softplus_input=(
                    alpha.astype(paddle.float32) + self.dt_bias.astype(paddle.float32)
                ),
                g=g,
                beta=beta,
                core_output=core_attn_out,
            )
        nvtx_range_pop(suffix="gated_delta_rule")

        # Gated norm
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # Reshape: [b, s, num_heads, head_dim] -> [b, s, v_dim]
        norm_out = norm_out.reshape([batch, seq_len, -1])

        # Transpose back if sequence parallel: [b, s, x] -> [s, b, x]
        if self.config.sequence_parallel:
            norm_out = norm_out.transpose([1, 0, 2]).contiguous()

        # Output projection
        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        """Apply output normalization with gating."""
        # x: [b, s, num_heads, head_dim], gate: [b, s, num_heads, head_dim]
        x_dtype = x.dtype
        x = x.reshape([-1, x.shape[-1]])
        y = self.out_norm(x)
        # Output gate
        gate = gate.reshape([-1, gate.shape[-1]])
        y = y * self.act_fn(gate.astype(paddle.float32))
        y = y.astype(x_dtype)
        return y

    def sharded_state_dict(self, structured_name_prefix: str = ""):
        """Provide a sharded state dictionary for distributed checkpointing."""
        try:
            from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
                build_sharded_state_dict,
            )
        except ImportError:
            return {}

        sharded_sd = {}

        # in_proj (ColumnParallelLinear) — delegate to its own sharded_state_dict
        if hasattr(self.in_proj, "sharded_state_dict"):
            sharded_sd.update(
                self.in_proj.sharded_state_dict(
                    structured_name_prefix=f"{structured_name_prefix}in_proj."
                )
            )

        # conv1d — TP-sharded along axis 0
        conv_sd = self.conv1d.state_dict(structured_name_prefix="")
        conv_shard_rules = None
        if self.tp_size > 1:
            conv_shard_rules = {"weight": 0}
            if self.conv_bias and "bias" in conv_sd:
                conv_shard_rules["bias"] = 0
        sharded_sd.update(
            build_sharded_state_dict(
                conv_sd,
                conv_shard_rules,
                f"{structured_name_prefix}conv1d.",
            )
        )

        # dt_bias and A_log — TP-sharded along axis 0
        param_sd = {"dt_bias": self.dt_bias, "A_log": self.A_log}
        param_shard_rules = None
        if self.tp_size > 1:
            param_shard_rules = {"dt_bias": 0, "A_log": 0}
        sharded_sd.update(
            build_sharded_state_dict(
                param_sd,
                param_shard_rules,
                structured_name_prefix,
            )
        )

        # out_norm — not TP-sharded (per-head norm)
        if hasattr(self.out_norm, "sharded_state_dict"):
            sharded_sd.update(
                self.out_norm.sharded_state_dict(
                    structured_name_prefix=f"{structured_name_prefix}out_norm."
                )
            )
        else:
            out_norm_sd = self.out_norm.state_dict(structured_name_prefix="")
            sharded_sd.update(
                build_sharded_state_dict(
                    out_norm_sd,
                    None,
                    f"{structured_name_prefix}out_norm.",
                )
            )

        # out_proj (RowParallelLinear) — delegate to its own sharded_state_dict
        if hasattr(self.out_proj, "sharded_state_dict"):
            sharded_sd.update(
                self.out_proj.sharded_state_dict(
                    structured_name_prefix=f"{structured_name_prefix}out_proj."
                )
            )

        return sharded_sd

    # def backward_dw(self):
    #     """Execute weight gradient computation for all linear layers."""
    #     self._backward_in_proj()
    #     self._backward_out_proj()

    # def _backward_in_proj(self):
    #     """Computes weight gradients of input projection layer."""
    #     self.in_proj.backward_dw()

    # def _backward_out_proj(self):
    #     """Computes weight gradients of output projection layer."""
    #     self.out_proj.backward_dw()


def paddle_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_accuracy_compatible_cumsum=False,
):
    """
    Paddle-native implementation of chunked gated delta rule for deterministic mode.

    This is a direct port from Megatron-LM.

    Reference: https://github.com/huggingface/transformers/blob/144c8ce2809a2e21914017652700e1ecb450501e/
        src/transformers/models/qwen3_next/modeling_qwen3_next.py#L470-L547
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)

    # Convert to [b, num_heads, s, head_dim] and float32
    query, key, value, beta, g = [
        x.transpose([0, 2, 1, 3]).contiguous().astype(paddle.float32)
        if x.ndim == 4
        else x.transpose([0, 2, 1]).contiguous().astype(paddle.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    if pad_size > 0:
        query = F.pad(query, [0, 0, 0, pad_size])
        key = F.pad(key, [0, 0, 0, pad_size])
        value = F.pad(value, [0, 0, 0, pad_size])
        beta = F.pad(beta, [0, pad_size])
        g = F.pad(g, [0, pad_size])

    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to chunks: [b, h, num_chunks, chunk_size, dim]
    query, key, value, k_beta, v_beta = [
        x.reshape([x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape([g.shape[0], g.shape[1], -1, chunk_size])

    mask = paddle.triu(
        paddle.ones([chunk_size, chunk_size], dtype=paddle.bool), diagonal=0
    )

    # Chunk decay. Torch CUDA uses a Sklansky scan for this 64-wide axis.
    g = (
        _accuracy_compatible_sklansky_scan(g)
        if use_accuracy_compatible_cumsum
        else g.cumsum(axis=-1)
    )
    decay_mask = (
        (g.unsqueeze(-1) - g.unsqueeze(-2))
        .tril()
        .exp()
        .astype(paddle.float32)
        .tril()
    )

    # attn = -((k_beta @ key^T) * decay_mask), masked to lower triangular
    attn = -(
        (k_beta @ key.transpose([0, 1, 2, 4, 3])) * decay_mask
    ).masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + paddle.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        last_recurrent_state = paddle.zeros(
            [batch_size, num_heads, k_head_dim, v_head_dim],
            dtype=value.dtype,
        )
    else:
        last_recurrent_state = initial_state.astype(value.dtype)

    core_attn_out = paddle.zeros_like(value)

    mask = paddle.triu(
        paddle.ones([chunk_size, chunk_size], dtype=paddle.bool), diagonal=1
    )

    # For each chunk
    num_chunks = total_sequence_length // chunk_size
    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (
            q_i @ k_i.transpose([0, 1, 3, 2]) * decay_mask[:, :, i]
        ).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (
                k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
            ).transpose([0, 1, 3, 2])
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        [
            core_attn_out.shape[0],
            core_attn_out.shape[1],
            -1,
            core_attn_out.shape[-1],
        ]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = (
        core_attn_out.transpose([0, 2, 1, 3]).contiguous().astype(initial_dtype)
    )
    return core_attn_out, last_recurrent_state
