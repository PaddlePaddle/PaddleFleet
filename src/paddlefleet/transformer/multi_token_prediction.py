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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import (
    LayerSpec,
    ScheduleNode,
    build_spec_layer,
)
from paddle.distributed.fleet.utils import recompute
from paddle.distributed.fleet.utils.sequence_parallel_utils import (
    ScatterOp,
    mark_as_sequence_parallel_parameter,
)

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
from paddlefleet.tensor_parallel.random import get_cuda_rng_tracker
from paddlefleet.transformer.dw_overlap import deferrable_linear
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


# ============================================================================
# roll_tensor (Paddle port of MCore multi_token_prediction.roll_tensor, `8c4df6b07`)
#
# Semantics:
#   * shifts=-1 → left-shift by one along `dims`; new-in position filled with 0.
#   * cu_seqlens_q is None → standard paddle.roll (single sequence).
#   * cu_seqlens_q provided → per-document roll respecting doc boundaries;
#     the last token of every packed document is zeroed out and cannot leak
#     into the next document as an MTP target.
#   * cp_group is None or size==1 → non-CP path.
#   * cp_group.nranks > 1 with cu_seqlens_q → NotImplementedError; the
#     mirror-chunk (DualChunkSwap) CP variant is not implemented on this path
#     (CP is instead handled by extract_local_zigzag_chunks at the call site).
#
# Note: we consciously do NOT wrap cu_seqlens_q in an MCore-style
# PackedSeqParams dataclass. ernie5's model backend consumes doc boundaries
# via startend_row_indices (int32 tensor) throughout — introducing
# PackedSeqParams would (a) create a new data structure alien to ernie5,
# and (b) risk flipping attention kernels onto the THD path
# (qkv_format="thd") which is not what the ernie5 flashmask attention
# expects.
#
# Return contract mirrors MCore: (rolled_tensor, rolled_tensor.sum()). The
# second element is consumed by MultiTokenPredictionBlock.forward as
# `num_tokens` for per-token-loss averaging.
# ============================================================================


def _roll_tensor_packed_seq(
    tensor, shifts, dims, cu_seqlens_q, cp_group=None, pad_value=0
):
    """Per-doc left-shift using cu_seqlens_q.

    Only supports ``shifts=-1``. ``dims`` may be any single axis of ``tensor``
    (the seq axis — dim=1 for a [B, L, H] embedding, dim=-1 for a [B, L]
    tokens tensor, etc.). The boundary position of each packed document is
    filled with ``pad_value`` to prevent cross-doc leakage. Use ``pad_value=0``
    for embeddings / input_ids (default) and ``pad_value=ignored_index`` for
    labels so the boundary token is masked out of the loss instead of being
    trained as a real target.

    CP support is intentionally omitted here; see the outer `roll_tensor`
    guard which routes CP-enabled calls to `_roll_tensor_packed_seq_cp`.

    Args:
        tensor: input tensor to roll along the sequence dimension.
        shifts: must be -1 (single-token left shift).
        dims: axis along which cu_seqlens_q indexes; any single axis of
            ``tensor``.
        cu_seqlens_q: int32 tensor of shape ``[num_docs + 1]``; cumulative
            document lengths so that ``cu_seqlens_q[-1] == tensor.shape[dims]``.
        cp_group: paddle process group for context parallelism; unused in
            this non-CP branch.
    """
    if shifts != -1:
        raise ValueError(
            f"Packed sequence roll only supports single-token left shift (shifts=-1), "
            f"got shifts={shifts}."
        )
    ndim = tensor.dim()
    dim = dims if dims >= 0 else ndim + dims
    if not (0 <= dim < ndim):
        raise ValueError(f"dims={dims} out of range for tensor of ndim={ndim}.")
    if cu_seqlens_q is None:
        raise ValueError("cu_seqlens_q must not be None.")

    if isinstance(cu_seqlens_q, paddle.Tensor):
        cu_seqlens_np = cu_seqlens_q.numpy().tolist()
    else:
        cu_seqlens_np = list(cu_seqlens_q)

    def _select(t, s, e):
        idx = [slice(None)] * ndim
        idx[dim] = slice(s, e)
        return t[tuple(idx)]

    rolled = tensor.clone()
    num_docs = len(cu_seqlens_np) - 1
    for i in range(num_docs):
        s = int(cu_seqlens_np[i])
        e = int(cu_seqlens_np[i + 1])
        if e - s <= 0:
            continue
        seg = _select(tensor, s, e)
        rolled_seg = paddle.roll(seg, shifts=shifts, axis=dim)
        # Overwrite the last position that would otherwise cross the doc
        # boundary. ``pad_value=0`` reproduces the original multiplicative
        # zero-fill (autograd-safe for float embeddings); a non-zero
        # ``pad_value`` (e.g. ignored_index for labels) fills the boundary via
        # ``keep + pad*(1-keep)`` so it stays a functional (non-in-place) op.
        seg_len = e - s
        if seg_len >= 1:
            keep_shape = [1] * ndim
            keep_shape[dim] = seg_len
            keep_mask = paddle.ones(keep_shape, dtype=rolled_seg.dtype)
            last_idx = [slice(None)] * ndim
            last_idx[dim] = slice(seg_len - 1, seg_len)
            keep_mask[tuple(last_idx)] = 0
            if pad_value == 0:
                rolled_seg = rolled_seg * keep_mask
            else:
                rolled_seg = rolled_seg * keep_mask + pad_value * (
                    1 - keep_mask
                )
        # Assign back along the seq dim.
        target_idx = [slice(None)] * ndim
        target_idx[dim] = slice(s, e)
        rolled[tuple(target_idx)] = rolled_seg

    return rolled, rolled.sum()


def roll_tensor(
    tensor,
    shifts=-1,
    dims=-1,
    cp_group=None,
    cu_seqlens_q=None,
    pad_value=0,
):
    """Roll the tensor input along the given dimension(s).

    Paddle port of MCore ``multi_token_prediction.roll_tensor``. Used by the
    packed-doc MTP path (config.use_erndata=True) to shift
    input_ids / position_ids / labels / loss_mask by one token at each MTP
    depth, while respecting packed-document boundaries.

    CP handling deliberately differs from MCore. Under PaddleFleet's data
    layout, dist_data_loader.py broadcasts ``input_ids`` / ``labels`` /
    ``loss_mask`` via the ``cp_mp_parallel_group`` (see
    ``ernie5/src/datasets/dist_data_loader.py:133-135``), so every CP rank
    already holds a full-length ``[B, L]`` copy — no zigzag scatter happens
    before the model. Consequently rolling reduces to standard CP=1
    semantics on the full-length tensor, and callers should invoke
    ``extract_local_zigzag_chunks`` after ``roll_tensor`` to obtain their
    local slice before embedding / loss.

    Args:
        tensor: input tensor.
        shifts: currently only -1 (single-token left shift).
        dims: currently only the last dimension.
        cp_group: paddle process group for context parallelism. Accepted for
            API compatibility with MCore's signature but not used: under
            PaddleFleet's full-length CP layout, per-rank ``paddle.roll``
            already yields globally-correct output.
        cu_seqlens_q: int32 tensor of shape ``[num_docs + 1]`` describing
            packed-document boundaries. If provided, per-doc shift is
            applied. If ``None``, a single-sequence roll is done.
        pad_value: value written into the new-in position(s) created by the
            left shift — at each packed-doc boundary when ``cu_seqlens_q`` is
            given, otherwise at the last sequence position. Defaults to 0
            (embeddings / input_ids); pass ``ignored_index`` for labels so the
            boundary token is excluded from the loss.

    Returns:
        (rolled_tensor, rolled_tensor.sum()). The second value mirrors MCore's
        contract for downstream num_tokens accumulation.
    """
    # cp_group is intentionally unused (see docstring); reference it to keep
    # linters quiet without altering behavior.
    del cp_group

    # Packed sequence path — per-doc roll, boundary filled with ``pad_value``
    # by `_roll_tensor_packed_seq`. Correct on full-length tensors regardless
    # of CP size because doc boundaries are described globally.
    if cu_seqlens_q is not None:
        return _roll_tensor_packed_seq(
            tensor,
            shifts,
            dims,
            cu_seqlens_q,
            cp_group=None,
            pad_value=pad_value,
        )

    # Standard (non-packed) path — matches paddle.roll semantics plus a
    # ``pad_value`` fill at the new-in position.
    if shifts != -1:
        raise ValueError(
            f"roll_tensor currently only supports shifts=-1, got shifts={shifts}."
        )
    ndim = tensor.dim()
    dim = dims if dims >= 0 else ndim + dims
    rolled = paddle.roll(tensor, shifts=shifts, axis=dims)
    seq_len = tensor.shape[dim]
    if seq_len >= 1:
        keep_mask_shape = [1] * ndim
        keep_mask_shape[dim] = seq_len
        keep_mask = paddle.ones(keep_mask_shape, dtype=rolled.dtype)
        idx = [slice(None)] * ndim
        idx[dim] = slice(seq_len - 1, seq_len)
        keep_mask[tuple(idx)] = 0
        if pad_value == 0:
            rolled = rolled * keep_mask
        else:
            rolled = rolled * keep_mask + pad_value * (1 - keep_mask)
    return rolled, rolled.sum()


def extract_local_zigzag_chunks(tensor_full, cp_rank, cp_size, axis=1):
    """Extract this CP rank's zigzag chunks from a full-length tensor.

    Mirrors PaddleFleet's ``scatter_balance`` layout
    (``context_parallel_utils.py:97-153``): each rank owns two chunks —

    * ``chunk_start = tensor_full[..., interval*r : interval*(r+1), ...]``
    * ``chunk_end   = tensor_full[..., L-interval*(r+1) : L-interval*r, ...]``

    concatenated along the seq axis.

    Extraction only — no CP communication. Use to obtain the local slice
    without invoking ``ContextParallelScatterOp``, which avoids the
    ``cp_size`` × embedding-lookup redundancy that would otherwise result
    from doing embedding on the full-length ``input_ids``.

    Callers under ``use_erndata=True`` typically:

    1. Roll the full-length int tensor with ``roll_tensor(cu_seqlens_q=...)``.
    2. Extract this rank's local slice via this helper.
    3. Feed the local slice to embedding / loss.

    Args:
        tensor_full: ``[..., L, ...]`` full-length tensor available on every rank.
        cp_rank: this rank's index within the CP group.
        cp_size: CP world size. ``cp_size == 1`` returns ``tensor_full`` unchanged.
        axis: sequence axis (default 1 for ``[B, L, ...]``).

    Returns:
        ``[..., L / cp_size, ...]`` tensor holding this rank's zigzag chunks.
    """
    if cp_size == 1:
        return tensor_full
    ndim = tensor_full.dim()
    dim = axis if axis >= 0 else ndim + axis
    seq_len = tensor_full.shape[dim]
    if seq_len % (cp_size * 2) != 0:
        raise ValueError(
            f"extract_local_zigzag_chunks: seq_len={seq_len} on axis={axis} "
            f"is not divisible by 2*cp_size={2 * cp_size}."
        )
    interval = seq_len // cp_size // 2
    chunk_start = paddle.slice(
        tensor_full,
        axes=[dim],
        starts=[interval * cp_rank],
        ends=[interval * (cp_rank + 1)],
    )
    chunk_end = paddle.slice(
        tensor_full,
        axes=[dim],
        starts=[seq_len - interval * (cp_rank + 1)],
        ends=[seq_len - interval * cp_rank],
    )
    return paddle.concat([chunk_start, chunk_end], axis=dim)


def build_startend_row_indices_from_cu_seqlens(
    cu_seqlens_q, batch_size, include_position_axis=False
):
    """Derive flashmask ``attn_mask_startend_row_indices`` from ``cu_seqlens_q``.

    The erndata MTP contract carries packed-document boundaries as a
    cumulative-length int32 vector ``[num_docs + 1]`` rather than a materialized
    attention mask. Every consumer of that contract (the main backbone via
    ``GPTEmbedding``, and each MTP depth via
    ``MultiTokenPredictionLayer._forward_megatron_style``) needs the equivalent
    flashmask boundaries, so the derivation lives here once.

    For a token at position ``i`` inside document ``j``
    (``cu[j] <= i < cu[j+1]``), the end row is ``cu[j+1]`` — i.e. attention is
    confined to the token's own document.

    Args:
        cu_seqlens_q: ``[num_docs + 1]`` int32 cumulative doc lengths. The last
            entry is the full sequence length ``L``.
        batch_size: batch axis size to broadcast to.
        include_position_axis: when True emit the 2-column
            ``[B, 1, L, 2]`` layout (``[end, position]``) that
            ``gpt_model_use_experimental_version`` expects; otherwise the
            1-column ``[B, 1, L, 1]`` fleet-mode layout.

    Returns:
        ``[batch_size, 1, L, 1 or 2]`` int32 tensor on GPU.
    """
    import numpy as _np

    cu_np = (
        cu_seqlens_q.numpy()
        if isinstance(cu_seqlens_q, paddle.Tensor)
        else _np.asarray(cu_seqlens_q)
    )
    seqlen = int(cu_np[-1])

    end = _np.zeros(seqlen, dtype=_np.int32)
    for j in range(len(cu_np) - 1):
        s, e = int(cu_np[j]), int(cu_np[j + 1])
        end[s:e] = e
    if include_position_axis:
        pos = _np.arange(seqlen, dtype=_np.int32)
        # [L, 2] -> [1, 1, L, 2]
        startend_np = _np.stack([end, pos], axis=-1)[None, None, ...]
    else:
        # [L] -> [1, 1, L, 1]
        startend_np = end[None, None, :, None]

    out = paddle.to_tensor(startend_np).cuda()
    if batch_size > 1:
        # expand is zero-copy; attention only reads these values.
        out = out.expand([batch_size, *out.shape[1:]])
    return out


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
            linear projection to be applied (non-mHC mode: [2h] -> [h]).
        e_proj (Union[LayerSpec, type]): Specification or instance of the
            embedding projection (mHC mode: [h] -> [h]).
        h_proj (Union[LayerSpec, type]): Specification or instance of the
            hidden state per-stream projection (mHC mode: [h] -> [h]).
        transformer_layer (Union[LayerSpec, type]): Specification
            or instance of the transformer block to be applied.
    """

    enorm: LayerSpec | type = None
    hnorm: LayerSpec | type = None
    eh_proj: LayerSpec | type = None
    e_proj: LayerSpec | type = None
    h_proj: LayerSpec | type = None
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

    submodules_kwargs = {
        "enorm": layer_norm_impl,
        "hnorm": layer_norm_impl,
        "transformer_layer": transformer_layer_spec,
        "layer_norm": layer_norm_impl,
    }

    if config.enable_hyper_connections:
        submodules_kwargs["e_proj"] = column_parallel_linear_impl
        submodules_kwargs["h_proj"] = column_parallel_linear_impl
    else:
        submodules_kwargs["eh_proj"] = column_parallel_linear_impl

    mtp_layer_spec = LayerSpec(
        layer=WeightOnlyMTPLayer
        if config.mtp_load_weight_only
        else MultiTokenPredictionLayer,
        sublayers_spec=MultiTokenPredictionLayerSublayersSpec(
            **submodules_kwargs
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

        self.mhc_enabled = config.enable_hyper_connections

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

        if self.mhc_enabled:
            # mHC mode: separate e_proj and h_proj, operating per-stream.
            # e_proj: [h] -> [h], applied to embedding then broadcast across streams.
            # h_proj: [h] -> [h], applied per-stream on hidden states.
            self.e_proj = build_spec_layer(
                self.sublayers_spec.e_proj,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            self.h_proj = build_spec_layer(
                self.sublayers_spec.h_proj,
                self.config.hidden_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            self.eh_proj = None

            # Learned contraction parameters for MTP output
            n = config.num_residual_streams
            hc_dim = config.hidden_size * n
            # learned_output_contract() computes in fp32; store the parameters
            # in fp32 too (Megatron marks hc_head_* keep_in_fp32).

            hc_param_dtype = "float32"
            self.hc_head_fn = self.create_parameter(
                shape=[hc_dim, n],
                dtype=hc_param_dtype,
                default_initializer=nn.initializer.Constant(0.0),
            )
            # Use model-parallel RNG tracker for Xavier init so that the
            # initialization is independent of pipeline layer_index.
            if paddle.distributed.get_world_size() <= 1:
                nn.initializer.XavierUniform()(self.hc_head_fn)
            else:
                with get_cuda_rng_tracker().fork():
                    nn.initializer.XavierUniform()(self.hc_head_fn)
            self.hc_head_base = self.create_parameter(
                shape=[n],
                dtype=hc_param_dtype,
                default_initializer=nn.initializer.Constant(0.0),
            )
            self.hc_head_scale = self.create_parameter(
                shape=[1],
                dtype=hc_param_dtype,
                default_initializer=nn.initializer.Constant(1.0),
            )
            self._cast_to_low_precision = False
            if self.sequence_parallel:
                self.hc_head_fn.is_distributed = False
                self.hc_head_base.is_distributed = False
                self.hc_head_scale.is_distributed = False
        else:
            # Non-mHC mode: eh_proj [2h] -> [h]
            # For the linear projection at the (k - 1)-th MTP layer, the input is the concatenation
            # of the i-th token's hidden states and the (i + K)-th token's decoder input,
            # so the input's shape is [s, b, 2*h].
            # The output will be sent to the following transformer layer,
            # so the output's shape should be [s, b, h].
            if self.config.gpt_model_use_experimental_version:
                self.eh_proj = paddle.incubate.nn.FusedLinear(
                    self.config.hidden_size * 2,
                    self.config.hidden_size,
                    bias_attr=self.config.use_bias,
                )
                if self.config.tensor_model_parallel_size > 1:
                    mark_as_sequence_parallel_parameter(self.eh_proj.weight)
                    if self.config.use_bias:
                        mark_as_sequence_parallel_parameter(self.eh_proj.bias)
            else:
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
            self.e_proj = None
            self.h_proj = None

        self.transformer_layer = build_spec_layer(
            self.sublayers_spec.transformer_layer,
            config=self.config,
            is_mtp_layer=True,
        )
        if not self.config.gpt_model_use_experimental_version:
            self.norm = build_spec_layer(
                self.sublayers_spec.layer_norm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.rms_norm_eps,
            )

        # MTP Magic Send: per-layer embedding and counter
        self.mtp_embed = None
        if config.enable_mtp_magic_send:
            import copy

            from paddlefleet.tensor_parallel import (
                VocabParallelEmbedding,
            )

            no_init_cfg = copy.copy(config)
            no_init_cfg.perform_initialization = False
            self.mtp_embed = VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                init_method=config.embedding_init_method,
                reduce_scatter_embeddings=False,
                config=no_init_cfg,
            )
            if self.config.context_parallel_size > 1:
                from paddlefleet.context_parallel_utils import (
                    mark_context_parallel_parameter_disable_scale_grad,
                )

                mark_context_parallel_parameter_disable_scale_grad(
                    self.mtp_embed
                )

            from paddlefleet.models.gpt.mtp_embedding_layer import (
                mtp_magic_instance,
            )

            self.magic_key = f"mtp_layer_{self.layer_number}"
            mtp_magic_instance.set_magic_count(self.magic_key, -1)

        self.offload_context = nullcontext()

    @property
    def transformer_layer_weights(self):
        return self.transformer_layer.named_parameters()

    def _concat_embeddings(
        self,
        hidden_states: paddle.Tensor,
        decoder_input: paddle.Tensor,
        mtp_hidden_inputs_mask: paddle.Tensor | None = None,
    ):
        """
        Concatenate the tokens before sending to transformer layer.

        In mHC mode, hidden_states is [s, b, n*h] (multi-stream) and decoder_input
        is [s, b, h] (single-stream embedding). Uses separate e_proj and h_proj.
        In non-mHC mode, concatenates and projects with eh_proj as before.
        """
        decoder_input = self.enorm(decoder_input)

        if self.mhc_enabled:
            # mHC mode: hidden_states is [s, b, n*h]
            n = self.config.num_residual_streams
            h = self.config.hidden_size
            s, b, _ = hidden_states.shape

            hs_streams = hidden_states.reshape([s, b, n, h])
            hs_streams = self.hnorm(hs_streams)

            # Apply mask if needed
            if mtp_hidden_inputs_mask is not None:
                # [B, 1, S] -> [B, S, 1]
                mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.transpose(
                    [0, 2, 1]
                ).astype(hs_streams.dtype)
                if (
                    get_context_parallel_world_size() > 1
                    and self.config.experimental_dataflow
                ):
                    mtp_hidden_inputs_mask = ContextParallelScatterOp.apply(
                        mtp_hidden_inputs_mask,
                        axis=1,
                        mode=self.config.cp_balance_mode,
                    )
                # when sp enable
                if self.sequence_parallel:
                    # [B, S/CP, 1] -> [S/CP, B, 1]
                    mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.transpose(
                        [1, 0, 2]
                    )
                    # [S/CP, B, 1] -> [S/CP/TP, B, 1]
                    mtp_hidden_inputs_mask = (
                        scatter_to_sequence_parallel_region(
                            mtp_hidden_inputs_mask
                        )
                    )
                hs_streams = hs_streams * mtp_hidden_inputs_mask.unsqueeze(-1)

            # e_proj: [.., h] -> [.., h/tp]
            e_out, _ = deferrable_linear(
                self.config, "mtp_e_proj", self.e_proj, decoder_input
            )
            # h_proj: applied per-stream [.., n, h] -> [.., n, h/tp]
            # 4D tensor [b,s,n,h] causes .t() error in backward; reshape to 3D first
            orig_shape = list(hs_streams.shape)  # [s/sp, b, n, h]
            if self.tensor_parallel > 1 and self.sequence_parallel:
                # [s/sp, b, n, h] --> [s, b, n, h]
                orig_shape[0] = orig_shape[0] * self.tensor_parallel
            hs_flat = hs_streams.reshape([-1, orig_shape[-1]])  # [s/sp*b*n, h]
            h_out, _ = deferrable_linear(
                self.config, "mtp_h_proj", self.h_proj, hs_flat
            )  # [s*b*n, h/tp]
            h_out = h_out.reshape([*orig_shape[:-1], -1])  # [s, b, n, h/tp]
            # Broadcast add before gather (saves one all-gather vs gathering separately)
            hidden_states = e_out.unsqueeze(-2) + h_out
            if self.tensor_parallel > 1:
                hidden_states = gather_from_tensor_model_parallel_region(
                    hidden_states
                )
            # Flatten back to [.., n*h]
            *leading, n, h = hidden_states.shape
            hidden_states = hidden_states.reshape([*leading, n * h])

            if self.sequence_parallel:
                hidden_states = scatter_to_sequence_parallel_region(
                    hidden_states
                )
        else:
            hidden_states = self.hnorm(hidden_states)
            # Apply mtp_hidden_inputs_mask to mask out hidden state contributions
            # at specific positions (e.g. EOS boundaries) in MTP.
            # mask shape: [B, 1, S] -> [B, S, 1] to broadcast with hidden_states [B, S, H]
            if mtp_hidden_inputs_mask is not None:
                mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.transpose(
                    [0, 2, 1]
                )
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
                        mtp_hidden_inputs_mask,
                        axis=1,
                        mode=self.config.cp_balance_mode,
                    )

                # when sp enable
                if self.sequence_parallel:
                    if self.config.gpt_model_use_experimental_version:
                        mtp_hidden_inputs_mask = mtp_hidden_inputs_mask.reshape(
                            [-1, 1]
                        )
                        mtp_hidden_inputs_mask = ScatterOp.apply(
                            mtp_hidden_inputs_mask
                        )
                    else:
                        # [B, S/CP, 1] -> [S/CP, B, 1]
                        mtp_hidden_inputs_mask = (
                            mtp_hidden_inputs_mask.transpose([1, 0, 2])
                        )
                        mtp_hidden_inputs_mask = (
                            scatter_to_sequence_parallel_region(
                                mtp_hidden_inputs_mask
                            )
                        )
                hidden_states = hidden_states * mtp_hidden_inputs_mask
            # At the (k - 1)-th MTP layer, concatenates the i-th token's hidden_states
            # and the (i + K)-th token's embedding, and combine them with linear projection.
            hidden_states = paddle.cat((decoder_input, hidden_states), -1)
            hidden_states = self.eh_proj(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states, _ = hidden_states
            # For tensor parallel we need to gather the tensor across the model-parallel
            # ranks after the linear projection. This used to call
            # `all_gather_last_dim_from_tensor_parallel_region`, but that utility reduces
            # the gradient in backward pass and was therefore incorrect in this context.
            # It has been replaced with the correct `gather_from_tensor_model_parallel_region`.
            if not self.config.gpt_model_use_experimental_version:
                if self.tensor_parallel > 1:
                    hidden_states = gather_from_tensor_model_parallel_region(
                        hidden_states
                    )
                # For sequence parallel, scatter after linear_fc and before transformer layer.
                if self.sequence_parallel:
                    hidden_states = scatter_to_sequence_parallel_region(
                        hidden_states
                    )
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
        swa_rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        swa_rotary_pos_cos: Tensor | None = None,
        swa_rotary_pos_sin: Tensor | None = None,
        attention_bias: paddle.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        attn_mask_startend_row_indices: paddle.Tensor | None = None,
        mtp_hidden_inputs_mask: paddle.Tensor | None = None,
        input_ids: paddle.Tensor | None = None,
        position_ids: paddle.Tensor | None = None,
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
                "swa_rotary_pos_emb": swa_rotary_pos_emb,
                "swa_rotary_pos_cos": swa_rotary_pos_cos,
                "swa_rotary_pos_sin": swa_rotary_pos_sin,
                "attention_bias": attention_bias,
                "packed_seq_params": packed_seq_params,
                "attn_mask_startend_row_indices": attn_mask_startend_row_indices,
                "is_mtp": True,
                "input_ids": input_ids,
                "position_ids": position_ids,
            }
            rst_dict = self.transformer_layer(input_dict)

        hidden_states = rst_dict["hidden_states"]

        # In mHC mode, skip postprocess here - it's deferred to forward()
        # so we can keep multi-stream state for subsequent MTP layers.
        if (
            not self.mhc_enabled
            and not self.config.gpt_model_use_experimental_version
        ):
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def _postprocess(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        """Postprocess MTP layer output: learned contraction in mHC mode + layer norm.

        In mHC mode, the hidden_states is multi-stream [s, b, n*h] and needs to be
        contracted to single-stream [s, b, h] before being used for loss computation.
        """
        if self.mhc_enabled:
            from paddlefleet.transformer.hyper_connection import (
                HyperConnectionModule,
            )

            hidden_states = HyperConnectionModule.learned_output_contract(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
                self.config.num_residual_streams,
                self.config.rms_norm_eps,
            )

        # Final layer norm
        if not self.config.gpt_model_use_experimental_version:
            hidden_states = self.norm(hidden_states)

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
            swa_rotary_pos_emb = kwargs.get("swa_rotary_pos_emb", None)
            swa_rotary_pos_cos = kwargs.get("swa_rotary_pos_cos", None)
            swa_rotary_pos_sin = kwargs.get("swa_rotary_pos_sin", None)
            attention_bias = kwargs.get("attention_bias", None)
            packed_seq_params = kwargs.get("packed_seq_params", None)
            mtp_hidden_inputs_mask = kwargs.get("mtp_hidden_inputs_mask", None)
            input_ids = kwargs.get("input_ids", None)
            position_ids = None
            if self.config.gpt_model_use_experimental_version:
                position_ids = kwargs.get("position_ids", None)
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
                swa_rotary_pos_emb=swa_rotary_pos_emb
                if swa_rotary_pos_emb is not None
                else None,
                swa_rotary_pos_cos=swa_rotary_pos_cos
                if swa_rotary_pos_cos is not None
                else None,
                swa_rotary_pos_sin=swa_rotary_pos_sin
                if swa_rotary_pos_sin is not None
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
                position_ids=position_ids if position_ids is not None else None,
            )

        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            assert self.config.recompute_num_layers == 1, (
                "recompute_num_layers must be 1 for MTP recompute"
            )
            outputs = checkpoint_handler()
        elif self.config.recompute_method in ("block", "first_n"):
            # "block" and "first_n" are decoder-layer concepts (based on
            # decoder layer_number vs recompute_num_layers).  They don't
            # apply to MTP layers, so skip recompute and run forward directly.
            outputs = forward_func(*args, **kwargs)
        else:
            raise ValueError("Invalid activation recompute method.")

        return outputs

    def forward(self, dict_args: dict):
        # Dispatch by config.use_erndata. Under erndata the data pipeline
        # emits no mtp_startend_row_indices_all / mtp_hidden_inputs_mask_all
        # and no L+K token concatenation; instead we shift input_ids /
        # position_ids / labels / loss_mask inside this layer via
        # roll_tensor(cu_seqlens_q=...).
        if getattr(self.config, "use_erndata", False):
            return self._forward_megatron_style(dict_args)

        if "context" in dict_args:
            assert dict_args["context"] is None, (
                "multi token prediction + cross attention is not yet supported."
            )
        if "packed_seq_params" in dict_args:
            assert dict_args["packed_seq_params"] is None, (
                "multi token prediction + sequence packing is not yet supported."
            )

        # === MTP input arrives outside hidden_states ===
        # hidden_states is the pure backbone output in both cases. The shifted MTP
        # embeddings either come from a local re-embedding of input_ids
        # (magic_send, PP > 1) or straight from GPTEmbedding through
        # mtp_decoder_inputs (separate_mtp_input, PP == 1), in which case they are
        # already CP/SP-scattered and must not be scattered again.
        if self.config.enable_mtp_magic_send or self.config.separate_mtp_input:
            prev = dict_args["hidden_states"]
            mhc_multistream = dict_args.pop("mhc_multistream", None)
            # Consumed by this layer only: pop it so it does not ride along in
            # the **kwargs passthrough of _proj_and_transformer_layer.
            mtp_decoder_inputs = dict_args.pop("mtp_decoder_inputs", None)
            if self.config.separate_mtp_input and mtp_decoder_inputs is None:
                raise RuntimeError(
                    "separate_mtp_input=True but mtp_decoder_inputs not found in "
                    "dict_args. GPTEmbedding may not have produced the shifted MTP "
                    "embeddings."
                )

            # Split prev into segments, take last as chain_input
            n_slices = self.layer_number + 1
            if self.layer_number == 0:
                prev_segs = [prev]
                chain_input = prev
            else:
                prev_segs = paddle.split(prev, n_slices)
                chain_input = prev_segs[-1]

            # mHC: split multi-stream into per-layer chunks, override chain_input
            mhc_enabled = mhc_multistream is not None
            mhc_chunks = None
            if mhc_enabled:
                mhc_chunks = paddle.split(
                    mhc_multistream, self.config.num_nextn_predict_layers + 1
                )
                chain_input = mhc_chunks[self.layer_number]

            # Main sequence length as seen from chain_input, which is already
            # CP-local and/or SP-local.  Used to trim rotary below, and (magic
            # send only) scaled back up to the global length for slicing the
            # re-embedded input_ids.
            cp_world_size = get_context_parallel_world_size()
            if self.config.sequence_parallel:
                seq_len = (
                    chain_input.shape[0]
                    * self.config.tensor_model_parallel_size
                )
            else:
                seq_len = chain_input.shape[1]
            depth = self.layer_number

            if self.config.separate_mtp_input:
                # GPTEmbedding already produced this depth's shifted embedding in
                # exactly the layout chain_input has, so there is nothing to
                # slice and nothing to scatter.
                decoder_input = mtp_decoder_inputs[depth]
                mtp_input_ids_all = dict_args.get(
                    "mtp_input_ids_for_moe_mask", None
                )
                mtp_input_ids_local = (
                    mtp_input_ids_all[:, depth, :].contiguous()
                    if mtp_input_ids_all is not None
                    else None
                )
            else:
                if cp_world_size > 1 and self.config.experimental_dataflow:
                    seq_len = seq_len * cp_world_size

                # --- Index-based input_ids addressing ---
                from paddlefleet.models.gpt.mtp_embedding_layer import (
                    mtp_magic_instance,
                )

                magic_count = mtp_magic_instance.get_magic_count(self.magic_key)
                # Skip increment during recompute replay
                if paddle.is_grad_enabled() or not self.training:
                    magic_count += 1
                    mtp_magic_instance.set_magic_count(
                        self.magic_key, magic_count
                    )
                input_ids_list = mtp_magic_instance.get("input_ids")
                magic_idx = magic_count % len(input_ids_list)
                input_ids = input_ids_list[magic_idx]

                # Re-embed input_ids locally
                mtp_input_embeds = self.mtp_embed(input_ids).astype(
                    self.mtp_embed.weight.dtype
                )

                # Zero-out padding for MoE routing
                if (
                    self.config.expert_model_parallel_size > 1
                    and self.config.tensor_model_parallel_size < 2
                ):
                    from paddlefleet.models.gpt.utils import fill_feature

                    pad_token_id = getattr(self.config, "pad_token_id", 0) or 0
                    mtp_input_embeds = fill_feature(
                        mtp_input_embeds, input_ids == pad_token_id, 0
                    )

                # Shifted embedding slice for current depth
                decoder_input = mtp_input_embeds[
                    :, (depth + 1) : (depth + 1 + seq_len), :
                ]

                # CP/SP scatter, mirroring what GPTEmbedding does per chunk
                if cp_world_size > 1 and self.config.experimental_dataflow:
                    decoder_input = ContextParallelScatterOp.apply(
                        decoder_input, axis=1, mode=self.config.cp_balance_mode
                    )
                if self.config.sequence_parallel:
                    batch_size, local_seq_len, hidden_size = decoder_input.shape
                    decoder_input = decoder_input.reshape(
                        [-1, decoder_input.shape[-1]]
                    )
                    decoder_input = ScatterOp.apply(decoder_input)
                    if not self.config.gpt_model_use_experimental_version:
                        decoder_input = (
                            decoder_input.reshape([batch_size, -1, hidden_size])
                            .permute(1, 0, 2)
                            .contiguous()
                        )  # [S/tp, B, H]

                # Per-depth input_ids for MoE mask
                mtp_input_ids_local = input_ids[
                    :, (depth + 1) : (depth + 1 + seq_len)
                ].contiguous()

            # Trim rotary embeddings to seq_len (once; seq_len is constant across depths)
            _rotary_keys = (
                "rotary_pos_emb",
                "rotary_pos_cos",
                "rotary_pos_sin",
                "swa_rotary_pos_emb",
                "swa_rotary_pos_cos",
                "swa_rotary_pos_sin",
            )
            for rk in _rotary_keys:
                rv = dict_args.get(rk, None)
                if rv is None:
                    continue
                if rk in ("rotary_pos_emb", "swa_rotary_pos_emb"):
                    dict_args[rk] = (
                        rv[:seq_len]
                        if self.config.sequence_parallel
                        else rv[:, :seq_len]
                    )
                else:
                    dict_args[rk] = rv[:, :seq_len]

            # Per-depth attention mask
            mtp_startend_row_indices_all = dict_args.get(
                "mtp_startend_row_indices_all", None
            )
            mtp_hidden_inputs_mask_all = dict_args.get(
                "mtp_hidden_inputs_mask_all", None
            )

            mtp_mask = None
            if mtp_startend_row_indices_all is not None:
                if self.config.gpt_model_use_experimental_version:
                    mtp_mask = mtp_startend_row_indices_all[
                        :, depth : depth + 1, :, :
                    ]
                else:
                    mtp_mask = mtp_startend_row_indices_all[
                        :, depth : depth + 1, :, :1
                    ]
            mtp_hidden_inputs_mask = (
                mtp_hidden_inputs_mask_all[:, depth : depth + 1, :]
                if mtp_hidden_inputs_mask_all is not None
                else None
            )

            # Update dict_args for _proj_and_transformer_layer call
            # (mirrors non-magic-send branch: update fields in dict_args, then **dict_args)
            dict_args["hidden_states"] = chain_input
            dict_args["decoder_input"] = decoder_input
            dict_args["attn_mask_startend_row_indices"] = mtp_mask
            dict_args["mtp_hidden_inputs_mask"] = mtp_hidden_inputs_mask
            dict_args["input_ids"] = mtp_input_ids_local
            # Remove keys not accepted by _proj_and_transformer_layer,
            # and also remove any None-valued keys (PP framework's
            # convert_tensor_dict_to_tuple crashes on None values).
            _pop_keys = (
                "mtp_startend_row_indices_all",
                "mtp_hidden_inputs_mask_all",
                "mhc_multistream",
                "labels",
                "mtp_input_ids_for_moe_mask",
            )
            _stashed = {}
            for k in _pop_keys:
                if k in dict_args:
                    _stashed[k] = dict_args.pop(k)
            # Remove None values to avoid PP framework crash
            _none_keys = [k for k, v in dict_args.items() if v is None]
            for k in _none_keys:
                dict_args.pop(k)

            # Projection + transformer
            if self.config.recompute_granularity == "full" and self.training:
                output = self._checkpointed_forward(
                    self._proj_and_transformer_layer,
                    **dict_args,
                )
            else:
                output = self._proj_and_transformer_layer(
                    **dict_args,
                )

            # Restore stashed keys back into dict_args
            dict_args.update(_stashed)

            # mHC: contract multi-stream to single-stream for concat
            if mhc_enabled:
                single_stream_output = self._postprocess(output)
            else:
                single_stream_output = output

            # Cumulative concat: append this layer's output
            new_hidden = paddle.concat([*prev_segs, single_stream_output])

            # Build return dict: only include tensors that need P2P communication.
            # PP framework serializes ALL dict values for send/recv, so we must
            # not include non-contiguous slices or unnecessary auxiliary tensors.
            new_args = {"hidden_states": new_hidden}
            for rk in (
                "rotary_pos_emb",
                "rotary_pos_cos",
                "rotary_pos_sin",
                "swa_rotary_pos_emb",
                "swa_rotary_pos_cos",
                "swa_rotary_pos_sin",
            ):
                val = dict_args.get(rk, None)
                if val is not None:
                    new_args[rk] = val
            if mtp_startend_row_indices_all is not None:
                new_args["mtp_startend_row_indices_all"] = (
                    mtp_startend_row_indices_all.contiguous()
                )
            if mtp_hidden_inputs_mask_all is not None:
                new_args["mtp_hidden_inputs_mask_all"] = (
                    mtp_hidden_inputs_mask_all.contiguous()
                )
            if "labels" in dict_args:
                new_args["labels"] = dict_args["labels"]
            if "input_ids" in dict_args:
                new_args["input_ids"] = dict_args["input_ids"]
            # Forward position_ids, attention_bias, blocks if present
            for extra_key in ("position_ids", "attention_bias", "blocks"):
                if extra_key in dict_args and dict_args[extra_key] is not None:
                    new_args[extra_key] = dict_args[extra_key]

            # mHC: pass multi-stream output to next MTP layer
            if (
                mhc_enabled
                and self.layer_number < self.config.num_nextn_predict_layers - 1
            ):
                mhc_chunks[self.layer_number + 1] = output
                new_args["mhc_multistream"] = paddle.concat(mhc_chunks)

            # Mark auxiliary tensors as stop_gradient for P2P
            _stop_grad_keys = (
                "mtp_startend_row_indices_all",
                "mtp_hidden_inputs_mask_all",
                "labels",
                "rotary_pos_emb",
                "rotary_pos_cos",
                "rotary_pos_sin",
                "swa_rotary_pos_emb",
                "swa_rotary_pos_cos",
                "swa_rotary_pos_sin",
            )
            for aux_key in _stop_grad_keys:
                val = new_args.get(aux_key, None)
                if val is not None and hasattr(val, "stop_gradient"):
                    val.stop_gradient = True

            return new_args

        # === Original concat+split logic ===
        hidden_states_concat = dict_args["hidden_states"]
        # mHC: pop multi-stream tensor if available
        mhc_multistream = dict_args.pop("mhc_multistream", None)

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

        # Split mhc_multistream chunks if available
        mhc_chunks = None
        if mhc_multistream is not None:
            mhc_chunks = paddle.split(
                mhc_multistream, self.config.num_nextn_predict_layers + 1
            )

        if self.config.train_mtp_only:
            for i in range(self.config.num_nextn_predict_layers):
                tensor_list = paddle.split(
                    hidden_states_concat,
                    self.config.num_nextn_predict_layers + 1,
                )
                if mhc_chunks is not None:
                    # mHC mode: use multi-stream as MTP input
                    dict_args["hidden_states"] = mhc_chunks[i]
                else:
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

                hidden_states = self._proj_and_transformer_layer(
                    **dict_args,
                )

                if mhc_chunks is not None:
                    # mHC: hidden_states is multi-stream, store for next depth
                    mhc_chunks[i + 1] = hidden_states
                    # Contract to single-stream for loss computation
                    tensor_list[i + 1] = self._postprocess(hidden_states)
                else:
                    tensor_list[i + 1] = hidden_states

                hidden_states_concat = paddle.concat(tensor_list)
            dict_args["hidden_states"] = hidden_states_concat
            dict_args.pop("decoder_input")
        else:
            tensor_list = paddle.split(
                hidden_states_concat, self.config.num_nextn_predict_layers + 1
            )
            if mhc_chunks is not None:
                # mHC mode: use multi-stream as MTP input
                dict_args["hidden_states"] = mhc_chunks[self.layer_number]
            else:
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

            hidden_states = self._proj_and_transformer_layer(
                **dict_args,
            )

            if mhc_chunks is not None:
                # mHC: hidden_states is multi-stream, store for next depth
                mhc_chunks[self.layer_number + 1] = hidden_states
                # Contract to single-stream for loss computation
                tensor_list[self.layer_number + 1] = self._postprocess(
                    hidden_states
                )
            else:
                tensor_list[self.layer_number + 1] = hidden_states

            hidden_states_concat = paddle.concat(tensor_list)
            dict_args["hidden_states"] = hidden_states_concat
            dict_args.pop("decoder_input")

        # mHC: pass updated multi-stream to subsequent MTP layers
        if (
            mhc_chunks is not None
            and self.layer_number < self.config.num_nextn_predict_layers - 1
        ):
            mhc_multistream = paddle.concat(mhc_chunks)
            dict_args["mhc_multistream"] = mhc_multistream

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

    # ------------------------------------------------------------------ #
    # Packed-doc MTP forward (config.use_erndata is True).
    #
    # Contract vs. the historical ernie5 path:
    #   * The data pipeline emits ONLY the main L-length tensors plus
    #     cu_seqlens_q. It does NOT emit
    #     mtp_startend_row_indices_all or mtp_hidden_inputs_mask_all,
    #     and it does NOT append K MTP tokens to input_ids / labels.
    #   * The shifted MTP embeddings for each depth are still prepared upstream
    #     by GPTEmbedding, but using roll_tensor(cu_seqlens_q=...) — i.e.
    #     per-doc left-shift with boundary zero-fill — rather than the L+K
    #     index-slice used by ernie5. GPTEmbedding.forward has a mirrored
    #     use_erndata branch that produces the same
    #     ``hidden_states_concat`` shape (concatenation of the main slice and
    #     K per-depth shifted slices), so the pipeline plumbing stays intact.
    #   * The MTP LMHead / loss layer downstream is responsible for rolling
    #     labels / loss_mask via roll_tensor before computing MTP loss (M2/M3
    #     will exercise this end-to-end).
    #
    # Behaviour inside this method for a given self.layer_number (=depth k):
    #   1. Split hidden_states_concat (shape [(K+1)*S, B, ...] under SP or
    #      [B, (K+1)*S, ...] otherwise) into (K+1) chunks along the sequence
    #      dim.
    #   2. Take chunk[k] as hidden_states (previous stage's output at this
    #      depth) and chunk[k+1] as decoder_input (upstream-computed shifted
    #      embedding at this depth).
    #   3. Run _proj_and_transformer_layer with:
    #        - mtp_hidden_inputs_mask=None (no per-doc EOS-derived hidden
    #          mask; per-doc boundary is expressed via packed_seq_params);
    #        - attn_mask_startend_row_indices=<main mask> (packed attn still
    #          consumes the main startend_row_indices — MTP layers inherit
    #          the same doc boundaries because they share cu_seqlens_q).
    #   4. Write the resulting hidden_states back into the chunk list at
    #      position k+1 and re-concat. Return the updated dict_args so the
    #      next MTP depth can consume its slice.
    #
    # Constraints: experimental_dataflow=False and enable_mtp_magic_send
    # disabled (both enforced at TransformerConfig.__post_init__). Context
    # parallelism is handled at the embedding / loss call sites via
    # extract_local_zigzag_chunks rather than inside roll_tensor.
    # ------------------------------------------------------------------ #

    def _forward_megatron_style(self, dict_args: dict) -> dict:
        # Cross-attention is still unsupported (identical constraint as
        # upstream MCore 8c4df6b07). Packed sequences ARE now supported —
        # that is the whole point of this branch — but they are represented
        # by a raw cu_seqlens_q int32 tensor in dict_args, NOT wrapped in
        # an MCore PackedSeqParams object (see the module-level docstring
        # for the rationale).
        if dict_args.get("context") is not None:
            raise NotImplementedError(
                "multi token prediction + cross attention is not yet supported "
                "under use_erndata=True."
            )
        if (
            dict_args.get("mtp_input_embeds") is not None
            or self.config.enable_mtp_magic_send
        ):
            # Config validation in TransformerConfig.__post_init__ should have
            # caught this, but keep a hard guard here for defence in depth.
            raise ValueError(
                "use_erndata=True is incompatible with enable_mtp_magic_send."
            )

        num_nextn = self.config.num_nextn_predict_layers

        hidden_states_concat = dict_args["hidden_states"]
        tensor_list = paddle.split(hidden_states_concat, num_nextn + 1)

        # Use previous-stage slices for this depth.
        dict_args["hidden_states"] = tensor_list[self.layer_number]
        dict_args["decoder_input"] = tensor_list[self.layer_number + 1]

        # Drop any leftover ernie5-path fields from dict_args so
        # _proj_and_transformer_layer sees the Megatron contract: no per-depth
        # hidden mask.
        dict_args.pop("mtp_hidden_inputs_mask", None)
        # If ernie5 upstream still smuggled these in, ignore them defensively.
        dict_args.pop("mtp_startend_row_indices_all", None)
        dict_args.pop("mtp_hidden_inputs_mask_all", None)
        dict_args.pop("mtp_input_ids_for_moe_mask", None)
        # input_ids: keep the main [B, L] slice for MoE routing mask etc.,
        # but do NOT slice it per-depth (there is only one canonical L now).
        input_ids = dict_args.get("input_ids")
        if input_ids is not None and input_ids.ndim > 2:
            # Defensive: some ernie5 codepaths stash [B, K, L] here.
            raise RuntimeError(
                f"Under use_erndata=True, input_ids must be [B, L], got shape {input_ids.shape}."
            )

        # Derive per-depth attn_mask_startend_row_indices from cu_seqlens_q.
        # Doc boundaries are the SAME at every MTP depth (per-doc roll does
        # not wrap across doc boundaries), so a single derivation is enough
        # per depth; we still recompute on each call in case the pipeline
        # dict has been mutated by an earlier depth.
        cu_seqlens_q = dict_args.get("cu_seqlens_q", None)
        if cu_seqlens_q is not None:
            # experimental_dataflow: 2-col [B, 1, S, 2]; fleet-mode: 1-col [B, 1, S, 1].
            # ernie5 flashmask & SWA helpers require a 4D layout [B, heads, S, num_vec]
            # (see startend_row_indices_add_sliding_window in utils.py).
            include_pos = bool(
                getattr(
                    self.config, "gpt_model_use_experimental_version", False
                )
            )
            batch_size = (
                dict_args["hidden_states"].shape[1]
                if getattr(self.config, "sequence_parallel", False)
                else dict_args["hidden_states"].shape[0]
            )
            dict_args["attn_mask_startend_row_indices"] = (
                build_startend_row_indices_from_cu_seqlens(
                    cu_seqlens_q,
                    batch_size,
                    include_position_axis=include_pos,
                )
            )

        # Run transformer layer.
        hidden_states = self._proj_and_transformer_layer(**dict_args)

        # Store back into the pipeline concat tensor at slot k+1.
        tensor_list[self.layer_number + 1] = hidden_states
        dict_args["hidden_states"] = paddle.concat(tensor_list)
        dict_args.pop("decoder_input", None)
        return dict_args


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
