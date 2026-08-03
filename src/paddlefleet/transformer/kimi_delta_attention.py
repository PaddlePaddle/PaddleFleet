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

# Kimi Delta Attention (KDA), the linear-attention layer of Kimi-Linear / Kimi-K3.
# Structure follows paddlefleet/transformer/gated_delta_net.py (GDN);
# semantics follow moonshotai Kimi modeling_kimi_linear.KimiDeltaAttention
# and fla.ops.kda.chunk_kda.
#
# Difference from GDN: the forget gate is per-channel (shape [b, s, hv, k]) instead
# of a per-head scalar, so the intra-chunk decay matrix cannot be written as an
# outer product and Akk / Aqk have to be built column by column.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    build_spec_layer,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    build_sharded_state_dict,
)

from paddlefleet.jit import jit_fuser
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.utils import (
    get_pg_size,
    nvtx_range_pop,
    nvtx_range_push,
)

from .gated_delta_net import _l2norm
from .paddle_norm import (
    get_norm_extra_args,
    mark_as_sequence_parallel_parameter,
)

if TYPE_CHECKING:
    from paddlefleet.transformer.transformer_config import TransformerConfig

try:
    from fla.ops.kda import chunk_kda

    HAVE_FLA = True
except ImportError:
    chunk_kda = None
    HAVE_FLA = False

logger = logging.getLogger(__name__)


@dataclass
class KimiDeltaAttentionSublayersSpec:
    """Layer specs for the projections and the output norm of KDA.

    in_proj / f_b_proj / g_b_proj are column parallel, out_proj is row parallel.
    f_a_proj / g_a_proj must be *replicated* (duplicated) linears: they are the
    low-rank bottleneck, so their full-rank output is what the column-parallel
    b_proj consumes, and under sequence parallel they run on the local sequence
    shard that b_proj then all-gathers.

    With use_full_rank_gate=True (what Kimi uses) the output gate is fused into
    in_proj and g_a_proj / g_b_proj are unused.
    """

    in_proj: LayerSpec | type = IdentityOp
    f_a_proj: LayerSpec | type = IdentityOp
    f_b_proj: LayerSpec | type = IdentityOp
    g_a_proj: LayerSpec | type = IdentityOp
    g_b_proj: LayerSpec | type = IdentityOp
    out_norm: LayerSpec | type = IdentityOp
    out_proj: LayerSpec | type = IdentityOp


class KimiDeltaAttention(FleetLayer):
    """Kimi Delta Attention (KDA) layer class.

    Takes input of size [b, s, h] (or [s, b, h] when sequence_parallel is enabled)
    and returns output of the same size.

    Only fixed-length (equal-length batch) input is supported for now; packed /
    variable-length sequences are not implemented.
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: KimiDeltaAttentionSublayersSpec,
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
        num_key_heads: int = 96,
        num_value_heads: int = 96,
        gate_lora_rank: int | None = None,
        use_full_rank_gate: bool = True,
        gate_lower_bound: float | None = -5.0,
    ):
        """
        Args:
            config: The transformer config of the model.
            sublayers_spec: Contains the layer specs for the projections and out_norm.
            layer_number: The layer number of this KDA layer.
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
            gate_lora_rank: Bottleneck rank of the forget-gate (and output-gate)
                low-rank projection. Defaults to value_head_dim, matching Kimi.
            use_full_rank_gate: Whether the output gate is a single full-rank
                projection (folded into in_proj) instead of a low-rank pair.
            gate_lower_bound: Lower bound of the log-space forget gate. When set,
                the gate becomes lower_bound * sigmoid(exp(A_log) * (a + dt_bias)),
                which is naturally clamped to [lower_bound, 0). Set to None to use
                -exp(A_log) * softplus(a + dt_bias) instead.
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
        self.use_full_rank_gate = use_full_rank_gate
        self.gate_lower_bound = gate_lower_bound

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp"]
            )
        assert pg_collection is not None, (
            "pg_collection must be provided for KimiDeltaAttention"
        )
        self.pg_collection = pg_collection
        self.tp_size = get_pg_size(self.pg_collection.tp)
        self.sp_size = self.tp_size if config.sequence_parallel else 1

        # Attributes from config
        self.hidden_size = config.hidden_size
        self.act_fn = config.hidden_act
        self.conv_kernel_dim = conv_kernel_dim
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.gate_lora_rank = (
            gate_lora_rank if gate_lora_rank is not None else value_head_dim
        )

        # q/k/v/beta are all sharded by head, so both head counts must divide
        # evenly; otherwise the per-tensor split sizes in forward() silently stop
        # adding up to the local in_proj width.
        for name, num_heads in (
            ("num_key_heads", self.num_key_heads),
            ("num_value_heads", self.num_value_heads),
        ):
            if num_heads % self.tp_size != 0:
                raise ValueError(
                    f"{name}({num_heads}) must be divisible by the tensor "
                    f"parallel size({self.tp_size})"
                )
        if self.num_value_heads % self.num_key_heads != 0:
            raise ValueError(
                f"num_value_heads({self.num_value_heads}) must be divisible by "
                f"num_key_heads({self.num_key_heads}) for GVA"
            )
        # The gate is per-channel over the key dim, but its projections are sized
        # by value_head_dim, so the two head dims have to match.
        if self.key_head_dim != self.value_head_dim:
            raise ValueError(
                f"KDA requires key_head_dim({self.key_head_dim}) == "
                f"value_head_dim({self.value_head_dim})"
            )

        # Input projection: q, k, v, beta and (if full rank) the output gate
        self.in_proj_dim = self.qk_dim * 2 + self.v_dim + self.num_value_heads
        if self.use_full_rank_gate:
            self.in_proj_dim += self.v_dim

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

        # Forget gate: low-rank hidden -> rank (replicated) -> v_dim (column parallel)
        self.f_a_proj, self.f_b_proj = self._build_lora_pair(
            sublayers_spec.f_a_proj, sublayers_spec.f_b_proj
        )
        # Output gate: either folded into in_proj (full rank) or another low-rank pair
        if not self.use_full_rank_gate:
            self.g_a_proj, self.g_b_proj = self._build_lora_pair(
                sublayers_spec.g_a_proj, sublayers_spec.g_b_proj
            )

        # Conv1D for QKV. Depthwise (groups=channels), so a single fused conv over
        # the concatenated q/k/v is exactly the three per-tensor short convolutions
        # used by the reference implementation.
        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv_dim_local_tp = self.conv_dim // self.tp_size

        # weight shape: [conv_dim, 1, d_conv], bias shape: [conv_dim]
        self.conv1d = nn.Conv1D(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            bias_attr=conv_bias,
            data_format="NCL",
        )
        self.conv1d.weight.is_distributed = True if self.tp_size > 1 else False
        if conv_bias and self.conv1d.bias is not None:
            self.conv1d.bias.is_distributed = (
                True if self.tp_size > 1 else False
            )

        self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
        self.v_dim_local_tp = self.v_dim // self.tp_size

        # dt_bias is per-channel for KDA (GDN has it per-head) — fp32 for softplus
        self.dt_bias = self.create_parameter(
            shape=[self.v_dim_local_tp],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.dt_bias.is_distributed = True if self.tp_size > 1 else False

        # A_log parameter — fp32 to avoid exp() overflow in bf16
        self.A_log = self.create_parameter(
            shape=[self.num_v_heads_local_tp],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.A_log.is_distributed = True if self.tp_size > 1 else False

        # Output layernorm before projection (per-head norm).
        # Replicated weight but each TP rank only sees its local value heads, so the
        # gradient is a partial sum and needs the sequence-parallel allreduce hook.
        input_is_parallel = True if self.tp_size > 1 else False
        extra_args = get_norm_extra_args(
            sublayers_spec.out_norm,
            self.config,
            self.value_head_dim,
            self.config.rms_norm_eps,
            input_is_parallel,
        )
        self.out_norm = build_spec_layer(sublayers_spec.out_norm, **extra_args)

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

    def _build_lora_pair(self, a_spec, b_spec):
        """hidden -> gate_lora_rank (replicated) -> v_dim (column parallel)."""
        a_proj = build_spec_layer(
            a_spec,
            self.hidden_size,
            self.gate_lora_rank,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )
        # b_proj is column parallel and takes the full rank as input, so a_proj
        # must not shard its output.
        a_out = getattr(
            a_proj, "output_size_per_partition", self.gate_lora_rank
        )
        if a_out != self.gate_lora_rank:
            raise ValueError(
                f"KDA gate a_proj must be a replicated linear, got "
                f"{type(a_proj).__name__} with an output sharded to {a_out} "
                f"instead of gate_lora_rank({self.gate_lora_rank})"
            )
        # The weight is replicated but each rank only feeds the local sequence
        # shard through it, so its gradient is a partial sum over tokens and
        # needs the same all-reduce hook as out_norm's weight.
        a_weight = getattr(a_proj, "weight", None)
        if self.config.sequence_parallel and a_weight is not None:
            mark_as_sequence_parallel_parameter(a_weight)

        b_proj = build_spec_layer(
            b_spec,
            self.gate_lora_rank,
            self.v_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )
        return a_proj, b_proj

    def reset_parameters(self):
        """Reset the parameters."""
        if self.config.perform_initialization:
            if self.conv_init is not None:
                nn.initializer.Uniform(
                    low=-self.conv_init, high=self.conv_init
                )(self.conv1d.weight)

            # dt_bias: 0 keeps the gate at its neutral point for both gate forms
            nn.initializer.Constant(0.0)(self.dt_bias)

            # A_log: initialize to log(uniform(A_init_range))
            A = paddle.empty([self.num_v_heads_local_tp], dtype="float32")
            nn.initializer.Uniform(
                low=self.A_init_range[0], high=self.A_init_range[1]
            )(A)
            paddle.assign(paddle.log(A), self.A_log)

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
        Perform a forward pass through the KDA module.

        Args:
            hidden_states: Hidden states [b, s, h] or [s, b, h] with sequence_parallel.
            attention_mask: Unused, KDA is causal by construction (only fixed-length
                input is supported, so there is no padding to mask out).
            key_value_states: Key/value states (for cross attention, not supported).
            attention_bias: Attention bias (unused).
            packed_seq_params: Parameters used for THD format (not supported).

        Returns:
            Tuple of (output, output_bias).
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "KDA does not support packed sequence for now."
            )

        hidden_states = hidden_states.contiguous()
        if self.config.sequence_parallel and self.sp_size > 1:
            # Input is [s, b, h] with sequence parallel
            seq_len_local, batch, _ = hidden_states.shape
            seq_len = seq_len_local * self.sp_size
        else:
            batch, seq_len, _ = hidden_states.shape

        # Input projection
        nvtx_range_push(suffix="in_proj")
        qkvbz, _ = self.in_proj(hidden_states)
        # Forget-gate logits: hidden -> rank -> v_dim, same input layout as in_proj
        alpha, _ = self.f_b_proj(self.f_a_proj(hidden_states)[0])
        if self.use_full_rank_gate:
            gate = None
        else:
            gate, _ = self.g_b_proj(self.g_a_proj(hidden_states)[0])
        nvtx_range_pop(suffix="in_proj")

        # Ensure [b, s, x] format for the rest of computation
        if self.config.sequence_parallel:
            qkvbz = qkvbz.transpose([1, 0, 2])
            alpha = alpha.transpose([1, 0, 2])
            if gate is not None:
                gate = gate.transpose([1, 0, 2])

        # Split into q, k, v, beta and (full-rank) output gate
        split_sizes = [
            self.qk_dim * 2 // self.tp_size,
            self.v_dim // self.tp_size,
            self.num_value_heads // self.tp_size,
        ]
        if self.use_full_rank_gate:
            split_sizes.append(self.v_dim // self.tp_size)
            qk, value, beta, gate = paddle.split(qkvbz, split_sizes, axis=-1)
        else:
            qk, value, beta = paddle.split(qkvbz, split_sizes, axis=-1)
        qkv = paddle.concat([qk, value], axis=-1)
        beta = beta.reshape([batch, seq_len, -1])
        gate = gate.reshape([batch, seq_len, -1, self.value_head_dim])
        alpha = alpha.reshape([batch, seq_len, -1, self.value_head_dim])

        # Convolution on qkv
        qkv = qkv.transpose([0, 2, 1]).contiguous()  # b, s, d -> b, d, s
        nvtx_range_push(suffix="conv1d")
        qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
        nvtx_range_pop(suffix="conv1d")

        qkv = qkv.transpose([0, 2, 1])  # b, d, s -> b, s, d
        query, key, value = paddle.split(
            qkv,
            [
                self.qk_dim // self.tp_size,
                self.qk_dim // self.tp_size,
                self.v_dim // self.tp_size,
            ],
            axis=-1,
        )
        query = query.reshape([batch, seq_len, -1, self.key_head_dim])
        key = key.reshape([batch, seq_len, -1, self.key_head_dim])
        value = value.reshape([batch, seq_len, -1, self.value_head_dim])

        if self.use_qk_l2norm:
            query = _l2norm(query.contiguous())
            key = _l2norm(key.contiguous())

        nvtx_range_push(suffix="kimi_delta_rule")
        if (not HAVE_FLA) or self.config.deterministic_mode:
            core_attn_out, _ = paddle_chunk_kda(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                g=alpha.contiguous(),
                beta=beta.contiguous(),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
            )
        else:
            raise NotImplementedError("FLA not supported yet.")
        nvtx_range_pop(suffix="kimi_delta_rule")

        # Gated norm
        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        # [b, s, num_heads, head_dim] -> [b, s, v_dim]
        norm_out = norm_out.reshape([batch, seq_len, -1])

        if self.config.sequence_parallel:
            norm_out = norm_out.transpose([1, 0, 2]).contiguous()

        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        return out, out_bias

    @jit_fuser
    def _apply_gated_norm(self, x, gate):
        """Per-head RMSNorm with a sigmoid output gate (KDA uses sigmoid, GDN silu)."""
        x_dtype = x.dtype
        x = x.reshape([-1, x.shape[-1]])
        y = self.out_norm(x)
        gate = gate.reshape([-1, gate.shape[-1]])
        y = y * F.sigmoid(gate.astype(paddle.float32))
        return y.astype(x_dtype)

    def sharded_state_dict(self, structured_name_prefix: str = ""):
        """Sharding along axis 0 for dt_bias / A_log and the depthwise conv1d.

        The parallel linears carry their own rules, and out_norm / the gate
        a_proj are replicated, so those are left to the recursive default.
        """
        sharded_state_dict = {}

        # dt_bias [v_dim / tp], A_log [num_value_heads / tp]
        own_sd = self.state_dict(
            structured_name_prefix="", include_sublayers=False
        )
        own_rules = None if self.tp_size == 1 else {"dt_bias": 0, "A_log": 0}
        sharded_state_dict.update(
            build_sharded_state_dict(own_sd, own_rules, structured_name_prefix)
        )

        for name, sublayer in self._sub_layers.items():
            if sublayer is None:
                continue
            prefix = f"{structured_name_prefix}{name}."
            if sublayer is self.conv1d:
                # Plain nn.Conv1D, so it has no rules of its own:
                # weight [conv_dim / tp, 1, kernel], bias [conv_dim / tp]
                conv_sd = sublayer.state_dict(structured_name_prefix="")
                conv_rules = None
                if self.tp_size > 1:
                    conv_rules = {"weight": 0}
                    if self.conv_bias:
                        conv_rules["bias"] = 0
                sharded_state_dict.update(
                    build_sharded_state_dict(conv_sd, conv_rules, prefix)
                )
            else:
                sharded_state_dict.update(
                    sublayer.sharded_state_dict(structured_name_prefix=prefix)
                )

        return sharded_state_dict


def kda_gate(g, A_log, dt_bias, safe_gate=False, lower_bound=None):
    """Raw gate logits [b, s, hv, k] -> log-space decay (fp32, always <= 0).

    Matches fla's fused gate:
        safe_gate=False: -exp(A_log) * softplus(g + dt_bias)
        safe_gate=True : lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
    """
    hv, k = g.shape[-2], g.shape[-1]
    x = g.astype(paddle.float32)
    if dt_bias is not None:
        x = x + dt_bias.astype(paddle.float32).reshape([hv, k])
    a = A_log.astype(paddle.float32).exp().reshape([hv, 1])
    if safe_gate:
        if lower_bound is None:
            raise ValueError("safe_gate=True requires lower_bound to be set")
        return lower_bound * F.sigmoid(a * x)
    return -a * F.softplus(x)


def _decay_bilinear(x, y, g):
    """Build A[..., c, j] = sum_d x[c,d] * exp(g[c,d] - g[j,d]) * y[j,d].

    x/y/g are [..., BT, D] and the result is [..., BT, BT]. Only the lower
    triangle (c >= j) is meaningful; the caller masks the rest.

    Clipping the exponent to <= 0 is required: g is a cumsum of non-positive
    values so the lower triangle already satisfies g[c] - g[j] <= 0 and the clip
    is a no-op there, but the upper triangle would overflow to inf. The forward
    pass hides that (those entries get masked to 0) while the backward pass
    turns inf into NaN.
    """
    BT = x.shape[-2]
    cols = []
    for j in range(BT):
        g_j = g[..., j : j + 1, :]
        y_j = y[..., j : j + 1, :]
        decay = paddle.clip(g - g_j, max=0.0).exp()
        cols.append(((x * decay) * y_j).sum(-1, keepdim=True))
    return paddle.concat(cols, axis=-1)


def _chunk_kda_core(query, key, value, g, beta, scale, initial_state, BT):
    """q/k: [b, s, h, k]; v: [b, s, hv, v]; g: [b, s, hv, k]; beta: [b, s, hv], fp32."""
    batch, seq_len, num_k_heads, k_head_dim = query.shape
    num_v_heads, v_head_dim = value.shape[2], value.shape[3]
    group = num_v_heads // num_k_heads

    # [b, s, h, x] -> [b, hv, s, x], q/k repeated to the value head count for GVA
    if group > 1:
        query = paddle.repeat_interleave(query, group, axis=2)
        key = paddle.repeat_interleave(key, group, axis=2)
    query = query.transpose([0, 2, 1, 3]) * scale
    key = key.transpose([0, 2, 1, 3])
    value = value.transpose([0, 2, 1, 3])
    g = g.transpose([0, 2, 1, 3])
    beta = beta.transpose([0, 2, 1])

    # Pad to a multiple of the chunk size. q/k/v/beta are zero padded and g is
    # padded with 0 (decay factor exp(0)=1), so beta=0 keeps the padded positions
    # from contributing to A/w/u/state and q=0 makes their output 0.
    pad_size = (BT - seq_len % BT) % BT
    if pad_size > 0:
        query = F.pad(query, [0, 0, 0, pad_size])
        key = F.pad(key, [0, 0, 0, pad_size])
        value = F.pad(value, [0, 0, 0, pad_size])
        g = F.pad(g, [0, 0, 0, pad_size])
        beta = F.pad(beta, [0, pad_size])
    total_seq_len = seq_len + pad_size
    num_chunks = total_seq_len // BT

    query, key, value, g = (
        x.reshape([batch, num_v_heads, num_chunks, BT, x.shape[-1]])
        for x in (query, key, value, g)
    )
    beta = beta.reshape([batch, num_v_heads, num_chunks, BT])
    g = g.cumsum(axis=-2)

    return _chunk_kda_scan(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        batch,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        seq_len,
        num_chunks,
        BT,
    )


def _chunk_kda_scan(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    batch,
    num_v_heads,
    k_head_dim,
    v_head_dim,
    seq_len,
    num_chunks,
    BT,
):
    """Intra-chunk WY representation plus the inter-chunk linear scan."""
    tri_incl = paddle.triu(paddle.ones([BT, BT], dtype="bool"), diagonal=0)
    tri_excl = paddle.triu(paddle.ones([BT, BT], dtype="bool"), diagonal=1)
    eye = paddle.eye(BT, dtype="float32")

    # Akk[c,j] = beta[c] * sum_d k[c,d] exp(g[c,d]-g[j,d]) k[j,d], strict lower
    attn = _decay_bilinear(key, key, g) * beta.unsqueeze(-1)
    attn = -attn.masked_fill(tri_incl, 0)
    # M = (I - attn)^{-1}. The reference does row-wise forward substitution with
    # setitem, which paddle rejects in backward (inplace version bump), and
    # paddle.linalg.triangular_solve has a wrong backward for the strict lower
    # triangle, so use inv() — I - attn is unit lower triangular, no pivoting.
    attn = paddle.linalg.inv(eye - attn)
    attn = attn * beta.unsqueeze(-2)

    w = attn @ (g.exp() * key)
    u = attn @ value

    if initial_state is None:
        state = paddle.zeros(
            [batch, num_v_heads, k_head_dim, v_head_dim], dtype="float32"
        )
    else:
        state = initial_state.astype("float32")

    outs = []
    for i in range(num_chunks):
        q_i, k_i, g_i = query[:, :, i], key[:, :, i], g[:, :, i]
        # Aqk[c,j] = sum_d q[c,d] exp(g[c,d]-g[j,d]) k[j,d], lower triangular
        attn_i = _decay_bilinear(q_i, k_i, g_i).masked_fill(tri_excl, 0)
        v_i = u[:, :, i] - w[:, :, i] @ state
        outs.append((q_i * g_i.exp()) @ state + attn_i @ v_i)
        g_last = g_i[:, :, -1]
        state = (
            state * g_last.unsqueeze(-1).exp()
            + ((g_last.unsqueeze(-2) - g_i).exp() * k_i).transpose([0, 1, 3, 2])
            @ v_i
        )

    core_attn_out = paddle.stack(outs, axis=2).reshape(
        [batch, num_v_heads, num_chunks * BT, v_head_dim]
    )[:, :, :seq_len]
    return core_attn_out.transpose([0, 2, 1, 3]), state


def paddle_chunk_kda(
    query,
    key,
    value,
    g,
    beta,
    scale: float | None = None,
    initial_state=None,
    output_final_state: bool = False,
    A_log=None,
    dt_bias=None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    state_v_first: bool = False,
    chunk_size: int = 64,
):
    """Paddle-native chunked KDA, aligned with fla.ops.kda.chunk_kda.

    The main body runs in fp32 and the output is cast back to value.dtype, so the
    result is at least as accurate as the fla kernel (which uses bf16 matmuls).

    Args:
        query, key: [b, s, h, k]; value: [b, s, hv, v]; g: [b, s, hv, k];
            beta: [b, s, hv]. GVA is applied when hv > h.
        scale: defaults to k ** -0.5.
        initial_state: [b, hv, k, v] fp32.
        A_log / dt_bias: required when use_gate_in_kernel is True; g is then the
            raw gate logits, otherwise g is the pre-computed log-space decay.
        state_v_first: return the final state as [b, hv, v, k] instead of
            [b, hv, k, v] (fla's transpose_state_layout).

    Only fixed-length input is supported; slice by cu_seqlens and call per
    sequence for variable-length data.

    Returns:
        (output, final_state). output has value's dtype, final_state is fp32 or
        None when output_final_state is False.
    """
    batch, seq_len, num_k_heads, k_head_dim = query.shape
    num_v_heads = value.shape[2]
    if num_v_heads % num_k_heads != 0:
        raise ValueError(
            f"num_v_heads({num_v_heads}) must be divisible by "
            f"num_k_heads({num_k_heads})"
        )
    expected_g = [batch, seq_len, num_v_heads, k_head_dim]
    if list(g.shape) != expected_g:
        raise ValueError(f"g must be {expected_g}, got {list(g.shape)}")
    expected_beta = [batch, seq_len, num_v_heads]
    if list(beta.shape) != expected_beta:
        raise ValueError(
            f"beta must be {expected_beta}, got {list(beta.shape)}"
        )
    if chunk_size not in (32, 64):
        raise ValueError(f"chunk_size must be 32 or 64, got {chunk_size}")
    if scale is None:
        scale = k_head_dim**-0.5

    out_dtype = value.dtype
    if use_qk_l2norm_in_kernel:
        query, key = _l2norm(query), _l2norm(key)
    beta_f = beta.astype("float32")
    if use_beta_sigmoid_in_kernel:
        beta_f = F.sigmoid(beta_f) * (2.0 if allow_neg_eigval else 1.0)
    if use_gate_in_kernel:
        g_f = kda_gate(g, A_log, dt_bias, safe_gate, lower_bound)
    else:
        g_f = g.astype("float32")

    core_attn_out, state = _chunk_kda_core(
        query.astype("float32"),
        key.astype("float32"),
        value.astype("float32"),
        g_f,
        beta_f,
        scale,
        initial_state,
        chunk_size,
    )
    if not output_final_state:
        return core_attn_out.astype(out_dtype), None
    if state_v_first:
        state = state.transpose([0, 1, 3, 2])
    return core_attn_out.astype(out_dtype), state
