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

import copy
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
from paddlefleet.recompute_utils import (
    need_recompute_in_block,
    need_recompute_in_first_n,
)
from paddlefleet.tensor_parallel import RecomputeWithoutOutput
from paddlefleet.transformer.identity_op import IdentityOp
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.utils import (
    get_pg_size,
    log_single_rank,
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
    from paddlefleet_ops import fla

    # Fused triton kernels from the paddle build of flash-linear-attention:
    # the short causal conv, the chunked KDA recurrence (with q/k L2 norm, the
    # gate and the beta sigmoid folded in) and the gated RMSNorm.
    causal_conv1d = fla.modules.conv.causal_conv1d
    chunk_kda = fla.ops.kda.chunk_kda
    rms_norm_gated = fla.modules.fused_norm_gate.rms_norm_gated
    build_cp_context = fla.ops.cp.context.build_cp_context

    HAVE_FLA = True
except (ImportError, AttributeError):
    causal_conv1d = chunk_kda = rms_norm_gated = build_cp_context = None
    HAVE_FLA = False

# Configure logging
logger = logging.getLogger(__name__)

# Every KDA layer of a model reaches the same verdict, so the backend is logged
# once per process instead of once per layer.
_FUSED_KERNEL_LOGGED = False


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


def build_cu_seqlens(
    startend_row_indices,
    batch,
    seq_len,
    keep_single_segment=False,
):
    """Derive packed cu_seqlens for a ``[b, s] -> [1, b*s]`` flattening.

    ``startend_row_indices`` is ``[b, 1, s, 1]`` and each entry holds the
    exclusive end of the document that position belongs to, *relative to its
    own row*. A position
    starts a new document iff that end changes, so the row-local boundaries
    plus the row seams (position 0 of every row) give exactly the segment
    starts of the flattened sequence. Note the seams matter: two adjacent
    rows routinely carry the same end value (e.g. both end at ``s``), so
    diffing the already-flattened array would merge them.

    Note ``paddlefleet.transformer.utils.get_doc_lens`` cannot be used here:
    it flattens across batch/head while comparing against a global arange,
    so it only produces correct lengths for ``[1, 1, s, 1]``.

    Every KDA layer of a step needs the same result, so the embedding builds it
    once and hands it down through ``dict_args["cu_seqlens"]``; KDA only
    calls this itself when nothing was passed in.

    With ``keep_single_segment`` the trivial ``[0, b*s]`` is returned instead
    of ``None``; context parallel always needs a cu_seqlens to slice.

    Returns the cu_seqlens tensor, or ``None`` when the whole flattened
    sequence is a single segment (nothing to mask out).
    """
    total = batch * seq_len
    if startend_row_indices is None:
        if not keep_single_segment:
            return None
        return paddle.to_tensor([0, total], dtype="int64")
    # The head dim must be 1: a linear recurrence has a single segment
    # layout, so a head-wise mask (what
    # startend_row_indices_add_sliding_window produces) cannot be
    # honoured and must not be silently reduced to head 0.
    if (
        startend_row_indices.ndim != 4
        or startend_row_indices.shape[1] != 1
        or startend_row_indices.shape[-1] != 1
    ):
        raise ValueError(
            "attn_mask_startend_row_indices must be [b, 1, s, 1], got "
            f"{list(startend_row_indices.shape)}"
        )
    if startend_row_indices.shape[-2] != seq_len:
        raise ValueError(
            f"attn_mask_startend_row_indices has sequence length "
            f"{startend_row_indices.shape[-2]}, expected {seq_len}"
        )
    if startend_row_indices.shape[0] != batch:
        raise ValueError(
            f"attn_mask_startend_row_indices has batch "
            f"{startend_row_indices.shape[0]}, expected {batch}"
        )
    # Column 0 is the exclusive document end.
    ends = startend_row_indices[:, 0, :, 0].astype("int64")

    # Fused Triton path: the same boundary detection + start compaction done in
    # a single kernel. Requires a CUDA-enabled build with an initialized Triton
    # driver AND the mask living on a GPU place; on a CPU build / CPU device we
    # must fall back to the pure-paddle implementation below.
    # NOTE: imported lazily (not at module top-level) on purpose. A module-level
    # `from paddlefleet.triton_ops.utils import is_triton_available` drags the whole
    # paddlefleet.triton_ops package into the paddlefleet import graph early, which
    # perturbs the transformers>=5.3 modeling_utils type-hint resolution order and
    # triggers `NameError: name 'Module' is not defined`. Keeping it function-local
    # preserves behavior without touching import ordering.
    from paddlefleet.triton_ops.utils import is_triton_available

    if is_triton_available() and ends.place.is_gpu_place():
        from paddlefleet.triton_ops.document_mask_fusion import (
            cu_seqlens_triton,
        )

        return cu_seqlens_triton(
            ends.flatten(), seq_len, keep_single_segment=keep_single_segment
        )

    doc_edges = ends[:, 1:] != ends[:, :-1]

    # Position 0 of every row is always a start, so the row seams become
    # segment boundaries too.
    head = paddle.ones([batch, 1], dtype="bool")
    if seq_len > 1:
        is_start = paddle.concat([head, doc_edges], axis=1)
    else:
        is_start = head
    # Flattened positions are ascending, so the flat indices of the starts
    # are already sorted and strictly increasing.
    starts = paddle.nonzero(is_start.reshape([-1])).flatten()
    cu_seqlens = paddle.concat(
        [starts, paddle.full([1], total, dtype=starts.dtype)]
    )
    if cu_seqlens.shape[0] <= 2 and not keep_single_segment:
        return None
    return cu_seqlens


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
        # Keep the parent-owned FP32 gate parameters (A_log and dt_bias) in
        # FP32 under AMP O2. Projection sublayers are decorated independently.
        self._cast_to_low_precision = False

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
        self.cp_size = get_pg_size(getattr(self.pg_collection, "cp", None))

        # Attributes from config
        self.hidden_size = config.hidden_size
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
        # Fused triton kernels for the conv / recurrence / gated norm. The paddle
        # native fallbacks stay in place for deterministic runs and for builds
        # without paddlefleet_ops.
        self.use_fused_kernels = HAVE_FLA and not config.deterministic_mode
        global _FUSED_KERNEL_LOGGED
        if self.use_fused_kernels and not _FUSED_KERNEL_LOGGED:
            _FUSED_KERNEL_LOGGED = True
            log_single_rank(logger, logging.INFO, "KDA will use fused kernel")

        # Selectively recompute the gated RMSNorm in backward instead of keeping
        # its output activation around. Uses RecomputeWithoutOutput so the norm
        # output buffer is discarded after the forward and rebuilt just before
        # out_proj's backward needs it. Honour the same layer-range semantics as
        # the other selective modules (Attention.core_attn / MLA.mla_qkv): a list
        # entry with recompute_num_layers, or a dict entry whose value is the
        # per-module layer count, restricts recompute to first_n / block layers.
        self.recompute_rms_norm_gated = False
        if self.config.recompute_granularity == "selective":
            modules = self.config.recompute_modules
            if isinstance(modules, list) and "rms_norm_gated" in modules:
                if self.config.recompute_num_layers is None:
                    self.recompute_rms_norm_gated = True
                else:
                    self.recompute_rms_norm_gated = self._need_recompute_layer(
                        self.config.recompute_num_layers
                    )
            elif isinstance(modules, dict) and "rms_norm_gated" in modules:
                self.recompute_rms_norm_gated = self._need_recompute_layer(
                    modules["rms_norm_gated"]
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

        # weight shape: [conv_dim, 1, d_conv], bias shape: [conv_dim].
        # fp32 like A_log / dt_bias / out_norm: the fused kernels accept an fp32
        # weight with low-precision activations.
        self.conv1d = nn.Conv1D(
            in_channels=self.conv_dim_local_tp,
            out_channels=self.conv_dim_local_tp,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim_local_tp,
            padding=self.conv_kernel_dim - 1,
            bias_attr=conv_bias,
            data_format="NCL",
            dtype="float32",
        )
        # force keep in float32 when using amp
        self.conv1d._cast_to_low_precision = False
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

        # force keep dt_bias / A_log (this layer's own params) in float32 under amp
        self._cast_to_low_precision = False

        # Output layernorm before projection (per-head norm).
        # Replicated weight but each TP rank only sees its local value heads, so the
        # gradient is a partial sum and needs the sequence-parallel allreduce hook.
        # The norm spec sizes its weight from config.params_dtype, so hand it a
        # config copy pinned to fp32.
        input_is_parallel = True if self.tp_size > 1 else False
        norm_config = copy.copy(self.config)
        norm_config.params_dtype = paddle.float32
        extra_args = get_norm_extra_args(
            sublayers_spec.out_norm,
            norm_config,
            self.value_head_dim,
            self.config.rms_norm_eps,
            input_is_parallel,
        )
        self.out_norm = build_spec_layer(sublayers_spec.out_norm, **extra_args)
        # force keep in float32 when using amp
        self.out_norm._cast_to_low_precision = False

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

    def _need_recompute_layer(self, recompute_num_layers):
        """Whether this layer_number is in the selective-recompute set.

        Mirrors Attention/MLA: recompute_method picks first_n vs block, and
        recompute_num_layers is the per-module layer count. Uses an explicit
        raise (not assert) so an invalid method cannot silently fall through to
        first_n under ``python -O``, which would change the recompute set.
        """
        if self.config.recompute_method == "block":
            return need_recompute_in_block(
                self.layer_number, self.config, recompute_num_layers
            )
        if self.config.recompute_method == "first_n":
            return need_recompute_in_first_n(
                self.layer_number, self.config, recompute_num_layers
            )
        raise ValueError(
            "selective recompute of rms_norm_gated with a layer count requires "
            "recompute_method to be 'first_n' or 'block', got "
            f"{self.config.recompute_method!r}"
        )

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
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        attention_mask: paddle.Tensor | None = None,
        key_value_states: paddle.Tensor | None = None,
        attention_bias: paddle.Tensor | None = None,
        packed_seq_params=None,
        input_ids: paddle.Tensor | None = None,
        cu_seqlens: paddle.Tensor | None = None,
        past_key_values=None,
        layer_idx: int | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[paddle.Tensor, paddle.Tensor | None]:
        """
        Perform a forward pass through the KDA module.

        Args:
            hidden_states: Hidden states [b, s, h] or [s, b, h] with sequence_parallel.
            attn_mask_startend_row_indices: [b, 1, s, 1] document boundaries. When
                given, the batch is flattened to a single packed sequence of length
                b*s and the resulting cu_seqlens is handed to both the short conv
                and the recurrence, so nothing leaks across document boundaries.
            attention_mask: Unused, KDA is causal by construction.
            key_value_states: Key/value states (for cross attention, not supported).
            attention_bias: Attention bias (unused).
            packed_seq_params: Parameters used for THD format (not supported).
            cu_seqlens: Packed cu_seqlens built once per step by the
                embedding (see ``build_cu_seqlens``) and passed down through
                ``dict_args``. Built here when None.
            past_key_values: Inference cache. KDA stores a fixed-size recurrent
                state plus the short conv's sliding window in it (see
                ``DynamicKVCache.set_kda_state``) instead of a growing K/V.
            layer_idx: This layer's slot in ``past_key_values``.
            use_cache: Whether to read/write ``past_key_values``. With it unset
                the whole sequence is recomputed from scratch, which is the
                ground truth the cached decode path is compared against.

        Returns:
            Tuple of (output, output_bias).
        """
        if packed_seq_params is not None:
            raise NotImplementedError(
                "KDA does not support packed sequence for now."
            )

        # Inference cache: the first pass prefills the recurrent state, every
        # later pass consumes it one token at a time. Both branches need the
        # cache slot, so a layer_idx-less cache is treated as no cache at all.
        cache_active = (
            use_cache and past_key_values is not None and layer_idx is not None
        )
        if cache_active:
            self._check_decode_supported(
                cu_seqlens, attn_mask_startend_row_indices
            )
            if past_key_values.has_kda_state(layer_idx):
                return self._decode_step(
                    hidden_states, past_key_values, layer_idx
                )

        hidden_states = hidden_states.contiguous()
        if self.config.sequence_parallel and self.sp_size > 1:
            # Input is [s, b, h] with sequence parallel
            seq_len_local, batch, _ = hidden_states.shape
            seq_len = seq_len_local * self.sp_size
        else:
            batch, seq_len, _ = hidden_states.shape

        if self.cp_size > 1:
            # fla's CP assumes the sequence is split contiguously and evenly
            # (part_len = total // world_size, rank_start = part_len * rank),
            # which is exactly scatter_contiguous. Other balance modes reorder
            # tokens, so the rank ranges would not match.
            if batch != 1:
                raise NotImplementedError(
                    "KDA context parallel requires batch == 1 (the packed "
                    f"sequence must be one contiguous split), got {batch}"
                )
            if "contiguous" not in getattr(
                self.config, "cp_balance_mode", "dualchunk_allgather"
            ):
                raise NotImplementedError(
                    "KDA context parallel requires a contiguous cp_balance_mode,"
                    f" got {getattr(self.config, 'cp_balance_mode', None)!r}"
                )
            if not self.use_fused_kernels:
                raise NotImplementedError(
                    "KDA context parallel requires the fused kernels."
                )
        # The mask always covers the full sequence, while hidden_states is this
        # rank's shard, so cu_seqlens has to be built in global coordinates.
        seq_len_full = seq_len * self.cp_size

        # Normally the embedding built this once for the whole step and it came
        # down through dict_args; only build it here when nothing was handed in.
        if cu_seqlens is None:
            cu_seqlens = build_cu_seqlens(
                attn_mask_startend_row_indices,
                batch,
                seq_len_full,
                keep_single_segment=self.cp_size > 1,
            )
        # The host copy is only a hint that lets the kernels skip a device-to-host
        # copy of their own. Leaving it None keeps every layer passing identical
        # arguments, so fla's @tensor_cache helpers (prepare_chunk_indices,
        # get_cp_cu_seqlens) hit on the shared cu_seqlens instead of recomputing.
        cu_seqlens_cpu = None
        if cu_seqlens is not None and not self.use_fused_kernels:
            raise NotImplementedError(
                "Variable-length input with document boundaries requires the "
                "fused kernels; the paddle native fallback has no cu_seqlens "
                "support."
            )
        cp_context = None
        if self.cp_size > 1:
            # build_cp_context slices the global cu_seqlens down to this rank and
            # records how many conv tokens / recurrent states to pull from the
            # neighbours. Both kernels read cu_seqlens out of the context and
            # ignore the argument, so drop the global one here.
            cp_context = build_cp_context(
                cu_seqlens,
                self.pg_collection.cp,
                conv1d_kernel_size=self.conv_kernel_dim,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
            cu_seqlens = cu_seqlens_cpu = None
        # Variable length is expressed as one packed sequence, which is what the
        # kernels require (they reject cu_seqlens with batch > 1). Under CP the
        # local shard is already [1, s_local], so nothing is flattened.
        eff_batch, eff_seq = (
            (1, batch * seq_len) if cu_seqlens is not None else (batch, seq_len)
        )

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

        # Split into q/k/v, beta and (full-rank) output gate. in_proj lays the
        # channels out as [qk | v | beta | gate] (see self.in_proj_dim) and the
        # conv wants q/k/v as one contiguous block, so keep them in a single
        # split segment rather than cutting qk/v apart and concatenating them
        # back together.
        split_sizes = [
            self.conv_dim_local_tp,
            self.num_value_heads // self.tp_size,
        ]
        if self.use_full_rank_gate:
            split_sizes.append(self.v_dim // self.tp_size)
            qkv, beta, gate = paddle.split(qkvbz, split_sizes, axis=-1)
        else:
            qkv, beta = paddle.split(qkvbz, split_sizes, axis=-1)
        qkv = qkv.reshape([eff_batch, eff_seq, -1])
        # The official router-strength projection is promoted before the
        # in-kernel sigmoid; keeping BF16 here changes the KDA state update.
        beta = beta.astype(paddle.float32).reshape([eff_batch, eff_seq, -1])
        gate = gate.reshape([eff_batch, eff_seq, -1, self.value_head_dim])
        alpha = alpha.reshape([eff_batch, eff_seq, -1, self.value_head_dim])

        # Prefill: snapshot the trailing conv window of the *pre-conv* qkv. The
        # depthwise conv of the next token needs the kernel_dim-1 raw inputs
        # before it, and nothing downstream of the conv can reconstruct them.
        # Zero-left-padded when the prompt is shorter than the window, which is
        # exactly what nn.Conv1D's causal padding contributed here.
        conv_state = None
        if cache_active:
            window = self.conv_kernel_dim - 1
            conv_state = paddle.zeros(
                [eff_batch, qkv.shape[-1], window], dtype=paddle.float32
            )
            if window > 0:
                kept = min(window, eff_seq)
                conv_state[..., window - kept :] = (
                    qkv[:, eff_seq - kept :, :]
                    .transpose([0, 2, 1])
                    .astype(paddle.float32)
                )

        # Convolution on qkv
        nvtx_range_push(suffix="conv1d")
        if self.use_fused_kernels:
            # Takes/returns [b, s, d] and fuses the causal padding, the depthwise
            # conv and the activation. Weight is [d, w] instead of [d, 1, w].
            # cu_seqlens is mandatory for packed input: without it the depthwise
            # conv pulls kernel_size-1 tokens across every document boundary.
            qkv, _ = causal_conv1d(
                qkv.contiguous(),
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation="silu",
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                cp_context=cp_context,
            )
        else:
            # nn.Conv1D needs matching dtypes, and the weight is pinned to fp32,
            # so run the fallback conv in fp32 and cast back.
            qkv_dtype = qkv.dtype
            qkv = qkv.transpose([0, 2, 1]).contiguous()  # b, s, d -> b, d, s
            qkv = self.conv1d(qkv.astype(self.conv1d.weight.dtype))
            qkv = F.silu(qkv[..., :eff_seq]).astype(qkv_dtype)
            qkv = qkv.transpose([0, 2, 1])  # b, d, s -> b, s, d
        nvtx_range_pop(suffix="conv1d")

        query, key, value = paddle.split(
            qkv,
            [
                self.qk_dim // self.tp_size,
                self.qk_dim // self.tp_size,
                self.v_dim // self.tp_size,
            ],
            axis=-1,
        )
        query = query.reshape([eff_batch, eff_seq, -1, self.key_head_dim])
        key = key.reshape([eff_batch, eff_seq, -1, self.key_head_dim])
        value = value.reshape([eff_batch, eff_seq, -1, self.value_head_dim])

        nvtx_range_push(suffix="kimi_delta_rule")
        if self.use_fused_kernels and not cache_active:
            # L2 norm, the log-space gate and sigmoid(beta) are all folded into
            # the kernel, matching the reference implementation.
            core_attn_out, _ = chunk_kda(
                q=query.contiguous(),
                k=key.contiguous(),
                v=value.contiguous(),
                g=alpha.contiguous(),
                beta=beta.contiguous(),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                cp_context=cp_context,
            )
        else:
            if self.use_qk_l2norm:
                query = _l2norm(query.contiguous())
                key = _l2norm(key.contiguous())
            core_attn_out, recurrent_state = paddle_chunk_kda(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                g=alpha.contiguous(),
                beta=beta.contiguous(),
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                use_qk_l2norm_in_kernel=False,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
                output_final_state=cache_active,
            )
        nvtx_range_pop(suffix="kimi_delta_rule")

        if cache_active:
            # Seed the decode loop. paddle_chunk_kda's final state is already the
            # fp32 [b, hv, k, v] the single-step recurrence expects, so no
            # correction term is needed between the last prefill token and the
            # first decode step.
            past_key_values.set_kda_state(
                layer_idx,
                recurrent_state,
                conv_state,
                num_new_tokens=seq_len,
            )

        # Gated norm
        nvtx_range_push(suffix="gated_norm")
        gated_norm_recompute = None
        if self.recompute_rms_norm_gated and self.training:
            # Drop the norm output activation now and re-run the gated norm
            # in backward. The recompute hook is registered on out_proj's
            # output below, so it fires right before out_proj needs its input.
            # The reshape/transpose are folded into the recomputed function so
            # the discarded tensor is exactly the one out_proj saves; otherwise
            # out_proj would hold a separate reshape view and nothing is freed.
            gated_norm_recompute = RecomputeWithoutOutput()
            norm_out = gated_norm_recompute.recompute(
                lambda c, g: self._gated_norm(c, g, batch, seq_len),
                core_attn_out,
                gate,
                preserve_rng_state=False,
                share_grad_holder=True,
            )
        else:
            norm_out = self._gated_norm(core_attn_out, gate, batch, seq_len)
        nvtx_range_pop(suffix="gated_norm")

        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")

        if gated_norm_recompute is not None:
            gated_norm_recompute.discard_output_and_register_recompute(out)

        return out, out_bias

    def _gated_norm(self, core_attn_out, gate, batch, seq_len):
        """Per-head gated RMSNorm, returning the [b, s, v_dim] output.

        Wraps both the fused FLA kernel and the paddle-native fallback, and
        folds in the final reshape (and the sequence-parallel transpose) so the
        whole gated-norm block is a single self-contained recompute segment.
        """
        if self.use_fused_kernels:
            norm_out = rms_norm_gated(
                core_attn_out.reshape([-1, self.value_head_dim]),
                gate.reshape([-1, self.value_head_dim]),
                self.out_norm.weight,
                None,
                activation="sigmoid",
                eps=self.config.rms_norm_eps,
            )
        else:
            norm_out = self._apply_gated_norm(core_attn_out, gate)

        # [b, s, num_heads, head_dim] -> [b, s, v_dim]
        norm_out = norm_out.reshape([batch, seq_len, -1])

        if self.config.sequence_parallel:
            norm_out = norm_out.transpose([1, 0, 2]).contiguous()
        return norm_out

    def _check_decode_supported(
        self, cu_seqlens, attn_mask_startend_row_indices
    ):
        """Reject what the paddle-native decode path does not cover yet.

        Raising here rather than in ``__init__`` keeps training unaffected: these
        features all work fine as long as the cache is not used.
        """
        if self.tp_size > 1:
            raise NotImplementedError(
                "KDA inference cache does not support tensor parallel yet."
            )
        if self.config.sequence_parallel:
            raise NotImplementedError(
                "KDA inference cache does not support sequence parallel yet."
            )
        if self.cp_size > 1:
            raise NotImplementedError(
                "KDA inference cache does not support context parallel yet."
            )
        if cu_seqlens is not None or attn_mask_startend_row_indices is not None:
            raise NotImplementedError(
                "KDA inference cache does not support variable-length "
                "(packed) input yet."
            )

    def _decode_step(self, hidden_states, past_key_values, layer_idx):
        """Advance one token using the cached recurrent state and conv window.

        This is the incremental form of the whole forward: the recurrence is
        unrolled for a single step in fp32 so it reproduces
        ``paddle_chunk_kda``'s scan exactly, and the short conv is evaluated
        against the cached window instead of the (no longer available) prefix.
        """
        batch, seq_len, _ = hidden_states.shape
        if seq_len != 1:
            raise NotImplementedError(
                "KDA decode consumes one token per step, got "
                f"seq_len={seq_len}. Chunked prefill would need the conv window "
                "and the recurrent state to be advanced by a whole block."
            )
        state, conv_state = past_key_values.get_kda_state(layer_idx)

        qkvbz, _ = self.in_proj(hidden_states)
        alpha, _ = self.f_b_proj(self.f_a_proj(hidden_states)[0])
        # _check_decode_supported has already rejected tp_size > 1, so the
        # unsharded widths are the local ones here. Spelling them without the
        # "// tp_size" of the full-sequence path keeps the TP restriction stated
        # in exactly one place instead of half-implemented in two.
        split_sizes = [self.conv_dim, self.num_value_heads]
        if self.use_full_rank_gate:
            split_sizes.append(self.v_dim)
            qkv, beta, gate = paddle.split(qkvbz, split_sizes, axis=-1)
        else:
            qkv, beta = paddle.split(qkvbz, split_sizes, axis=-1)
            gate, _ = self.g_b_proj(self.g_a_proj(hidden_states)[0])

        # Depthwise conv over [cached window | new token]. nn.Conv1D with
        # padding=w-1 truncated to the input length computes
        #   y[t] = sum_i weight[i] * x[t - (w-1) + i]
        # so weight index 0 pairs with the *oldest* token: the window must be
        # ordered oldest -> newest, and it slides left by one afterwards.
        qkv_dtype = qkv.dtype
        window = paddle.concat(
            [
                conv_state,
                qkv.transpose([0, 2, 1]).astype(conv_state.dtype),
            ],
            axis=-1,
        )
        conv_out = (window * self.conv1d.weight.squeeze(1)).sum(-1)
        if self.conv1d.bias is not None:
            conv_out = conv_out + self.conv1d.bias
        qkv = F.silu(conv_out).astype(qkv_dtype).unsqueeze(1)

        query, key, value = paddle.split(
            qkv, [self.qk_dim, self.qk_dim, self.v_dim], axis=-1
        )
        query = query.reshape([batch, -1, self.key_head_dim])
        key = key.reshape([batch, -1, self.key_head_dim])
        value = value.reshape([batch, -1, self.value_head_dim])
        alpha = alpha.reshape([batch, 1, -1, self.value_head_dim])
        gate = gate.reshape([batch, 1, -1, self.value_head_dim])

        if self.use_qk_l2norm:
            query = _l2norm(query)
            key = _l2norm(key)
        g = kda_gate(
            alpha,
            self.A_log,
            self.dt_bias,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
        ).squeeze(1)
        beta = F.sigmoid(beta.reshape([batch, -1]).astype(paddle.float32))

        core_attn_out, state = paddle_recurrent_kda_step(
            query.astype(paddle.float32),
            key.astype(paddle.float32),
            value.astype(paddle.float32),
            g,
            beta,
            self.key_head_dim**-0.5,
            state,
        )
        past_key_values.set_kda_state(
            layer_idx, state, window[..., 1:], num_new_tokens=1
        )

        norm_out = self._apply_gated_norm(
            core_attn_out.astype(value.dtype), gate
        )
        norm_out = norm_out.reshape([batch, seq_len, -1])
        return self.out_proj(norm_out)

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


def paddle_recurrent_kda_step(query, key, value, g, beta, scale, state):
    """One step of the KDA recurrence, matching _naive_recurrent_kda exactly.

    Shapes:
        query, key: [b, h, k]     (h = num_key_heads before GVA)
        value:     [b, hv, v]
        g:         [b, hv, k]     already in log-space (post kda_gate)
        beta:      [b, hv]        already through sigmoid
        state:     [b, hv, k, v]  fp32, updated in place-of-return
    Returns: (out [b, hv, v], new_state [b, hv, k, v]).
    """
    num_v_heads = value.shape[1]
    num_k_heads = query.shape[1]
    group = num_v_heads // num_k_heads
    if group > 1:
        query = paddle.repeat_interleave(query, group, axis=1)
        key = paddle.repeat_interleave(key, group, axis=1)
    query = query * scale
    # state[b, hv, k, v] *= exp(g)[b, hv, k, 1]
    state = state * g.unsqueeze(-1).exp()
    # delta[b, hv, v] = v - sum_k(k * state)
    delta = value - (key.unsqueeze(-1) * state).sum(-2)
    # state += (beta * k)[b, hv, k, 1] * delta[b, hv, 1, v]
    state = state + (beta.unsqueeze(-1) * key).unsqueeze(-1) * delta.unsqueeze(
        -2
    )
    # out[b, hv, v] = sum_k(q * state)
    out = (query.unsqueeze(-1) * state).sum(-2)
    return out, state


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
