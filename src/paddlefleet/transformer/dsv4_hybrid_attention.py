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

"""
DeepSeekV4 Hybrid Attention with Compressed Sparse Attention.

Ported from Megatron-LM experimental_attention_variant/deepseek_v4_hybrid_attention.py
(commit bf4e1db).

Components:
  - DSv4HybridAttention: Base class with inverse RoPE, grouped output projection
  - DSv4HybridSelfAttention: Self-attention with Q low-rank, single-head KV
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer
from paddle.distributed.fleet.utils import recompute

from paddlefleet.fp8.qat import fp8_simulate_qat
from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from paddlefleet.recompute_utils import (
    keep_indexer_grad_path,
    module_needs_recompute,
)
from paddlefleet.tensor_parallel import RecomputeWithoutOutput
from paddlefleet.transformer.attention import Attention
from paddlefleet.transformer.csa_attention import (
    CSADocMaskMetadata,
)
from paddlefleet.transformer.dw_overlap import (
    deferrable_linear,
    deferred_grouped_dw_accumulator,
)

if TYPE_CHECKING:
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig

try:
    from paddlefleet_ops import deep_gemm

    _DEEP_GEMM_AVAILABLE = hasattr(deep_gemm, "fp8_einsum")
except (ImportError, RuntimeError):
    deep_gemm = None
    _DEEP_GEMM_AVAILABLE = False


def _fleet_fp8_wo_a_gemm_enabled():
    if os.environ.get("FLEET_FP8_WO_A_GEMM", "1") in ("0", "false", "False"):
        return False
    return (
        _DEEP_GEMM_AVAILABLE
        and paddle.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0
        and paddle.device.cuda.get_device_capability()[0] >= 10
    )


FLEET_FP8_WO_A_GEMM = _fleet_fp8_wo_a_gemm_enabled()


def _q_rms_norm(
    q: Tensor,
    eps: float,
    high_precision_norm: bool,
    use_fusion: bool = False,
) -> Tensor:
    """RMS normalization for query (no learnable weight)."""
    if high_precision_norm:
        ori_dtype = q.dtype
        q = q.float()
        q = q * paddle.rsqrt(q.square().mean(-1, keepdim=True) + eps)
        return q.astype(ori_dtype)
    else:
        if use_fusion:
            from paddlefleet.triton_ops import fused_q_rms_norm

            result = fused_q_rms_norm(q, eps=eps)
        else:
            result = q * paddle.rsqrt(q.square().mean(-1, keepdim=True) + eps)
        return result


def _quant_blockwise(x: Tensor, quant_method: str):
    """Blockwise FP8 quantization returning ``(fp8, pow2_scale)``.

    Note: ``fp8_einsum`` (used by ``GroupedOutputFP8``) does NOT support UE8M0
    (int32-packed) scales — it only accepts float32 per-block scales.  Therefore
    this helper always produces the standard pow2 float32 layout.
    """
    return paddle.incubate.nn.functional.fp8_quant_blockwise(
        x,
        output_scale_transpose=False,
        quant_method=quant_method,
        input_transpose=False,
        using_pow2_scale=True,
    )[:2]


class GroupedOutputFP8(paddle.autograd.PyLayer):
    """FP8 grouped output projection ``"...gd,grd->...gr"``.

    Runs the grouped GEMM through ``deep_gemm.fp8_einsum`` with blockwise
    quantization (1x128 for activations/gradients, 128x128 for weights).
    ``dgrad`` is always FP8; ``wgrad`` is FP8 unless ``fp8_wgrad=False``.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        num_groups: int,
        o_lora_rank: int,
        fp8_wgrad: bool = True,
        save_original_input: bool = False,
    ):
        assert _DEEP_GEMM_AVAILABLE, (
            "GroupedOutputFP8 requires deep_gemm.fp8_einsum"
        )
        b, sq, _, d = x.shape
        assert d % 128 == 0, (
            "FP8 grouped output requires per-group hidden dim to be "
            f"divisible by 128, got {d}"
        )
        assert o_lora_rank % 128 == 0, (
            "FP8 grouped output requires o_lora_rank to be divisible by 128, "
            f"got {o_lora_rank}"
        )
        weight_bf16 = weight.reshape([num_groups, o_lora_rank, d])
        weight_for_gemm = weight_bf16.transpose([0, 2, 1]).contiguous()

        x_fp8, x_scale = _quant_blockwise(
            x.reshape([-1, d]).contiguous(), "1x128"
        )
        x_fp8 = x_fp8.reshape([b * sq, num_groups, d])
        x_scale = x_scale.reshape([b * sq, num_groups, -1])

        weight_fp8, weight_scale = _quant_blockwise(
            weight_for_gemm.reshape([-1, o_lora_rank]), "128x128"
        )
        weight_fp8 = weight_fp8.reshape([num_groups, d, o_lora_rank])
        weight_scale = weight_scale.reshape(
            [num_groups, d // 128, o_lora_rank // 128]
        )

        out = paddle.empty(
            [b * sq, num_groups, o_lora_rank], dtype=paddle.bfloat16
        )
        deep_gemm.fp8_einsum(
            "bhd,hdr->bhr",
            (x_fp8, x_scale),
            (weight_fp8, weight_scale),
            out,
            recipe=(1, 128, 128),
        )

        # ``save_original_input=False`` (default) skips stashing the bf16
        # activation to save memory: the FP8 wgrad path only needs the
        # "1x128" quantized activation, so it is produced eagerly here.
        if save_original_input:
            ctx.save_for_backward(x, weight_bf16)
            ctx.bf16_x_saved = True
            ctx.x_fp8_w_pre = None
            ctx.x_scale_w_pre = None
            ctx.x_shape = None
        else:
            assert (b * sq) % 128 == 0, (
                "GroupedOutputFP8 save_original_input=False requires "
                f"(batch*seqlen) divisible by 128, got b={b}, sq={sq}"
            )
            # [b*sq, h, d] -> [h, d, b*sq] so scales are along the b*sq axis.
            x_wgrad_pre = (
                x.reshape([b * sq, num_groups, d])
                .transpose([1, 2, 0])
                .contiguous()
            )
            x_fp8_w_pre, x_scale_w_pre = _quant_blockwise(
                x_wgrad_pre.reshape([-1, b * sq]), "1x128"
            )
            x_fp8_w_pre = x_fp8_w_pre.reshape([num_groups, d, b * sq])
            x_scale_w_pre = x_scale_w_pre.reshape([num_groups, d, -1])
            ctx.save_for_backward(weight_bf16)
            ctx.bf16_x_saved = False
            ctx.x_fp8_w_pre = x_fp8_w_pre
            ctx.x_scale_w_pre = x_scale_w_pre
            ctx.x_shape = (b, sq, num_groups, d)
        ctx.num_groups = num_groups
        ctx.o_lora_rank = o_lora_rank
        ctx.fp8_wgrad = fp8_wgrad
        # Paddle PyLayer requires None for stop_gradient inputs; record here so
        # a frozen backbone (phase 2 ``csa_train_indexer_only``) also skips the
        # wgrad GEMM instead of violating the contract.
        ctx.x_needs_grad = not x.stop_gradient
        ctx.weight_needs_grad = not weight.stop_gradient
        return out.reshape([b, sq, num_groups * o_lora_rank])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.bf16_x_saved:
            x, weight = ctx.saved_tensor()
            b, sq, _, d = x.shape
        else:
            (weight,) = ctx.saved_tensor()
            b, sq, _, d = ctx.x_shape
            x = None
        num_groups = ctx.num_groups
        o_lora_rank = ctx.o_lora_rank
        fp8_wgrad = ctx.fp8_wgrad
        grad_output = grad_output.reshape([b, sq, num_groups, o_lora_rank])

        # dgrad (always FP8): grad_x = grad_output @ weight, "bhr,hdr->bhd",
        # so weight [g, r, d] is transposed into the [g, d, r] "hdr" layout.
        go_fp8, go_scale = _quant_blockwise(
            grad_output.reshape([-1, o_lora_rank]).contiguous(), "1x128"
        )
        go_fp8 = go_fp8.reshape([b * sq, num_groups, o_lora_rank])
        go_scale = go_scale.reshape([b * sq, num_groups, -1])

        w_for_dgrad = weight.transpose([0, 2, 1]).contiguous()
        w_fp8, w_scale = _quant_blockwise(
            w_for_dgrad.reshape([-1, o_lora_rank]), "128x128"
        )
        w_fp8 = w_fp8.reshape([num_groups, d, o_lora_rank])
        w_scale = w_scale.reshape([num_groups, d // 128, o_lora_rank // 128])

        grad_x = paddle.empty([b * sq, num_groups, d], dtype=paddle.bfloat16)
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (go_fp8, go_scale),
            (w_fp8, w_scale),
            grad_x,
            recipe=(1, 128, 128),
        )
        grad_x = grad_x.reshape([b, sq, num_groups, d])

        if not ctx.weight_needs_grad:
            return (grad_x if ctx.x_needs_grad else None), None

        if fp8_wgrad:
            # grad_weight = x^T @ grad_output, "bhd,bhr->hdr". fp8_einsum
            # permutes both operands by {1, 2, 0}, and recipe=(1, 1, 128)
            # expects scales along the b*sq axis, so quantize on the
            # [h, {d,r}, b*sq] layout and permute back to [b*sq, h, {d,r}].
            assert (b * sq) % 128 == 0, (
                "FP8 grouped output wgrad requires (batch*seqlen) to be "
                f"divisible by 128, got b={b}, sq={sq}, b*sq={b * sq}"
            )
            if ctx.bf16_x_saved:
                x_wgrad = (
                    x.reshape([b * sq, num_groups, d])
                    .transpose([1, 2, 0])
                    .contiguous()
                )
                x_fp8_w, x_scale_w = _quant_blockwise(
                    x_wgrad.reshape([-1, b * sq]), "1x128"
                )
                x_fp8_w = x_fp8_w.reshape([num_groups, d, b * sq])
                x_scale_w = x_scale_w.reshape([num_groups, d, -1])
            else:
                # Reuse fp8 activation pre-computed in forward (identical
                # quantization to the bf16-saved path, so precision matches).
                x_fp8_w = ctx.x_fp8_w_pre
                x_scale_w = ctx.x_scale_w_pre

            go_wgrad = (
                grad_output.reshape([b * sq, num_groups, o_lora_rank])
                .transpose([1, 2, 0])
                .contiguous()
            )
            go_fp8_w, go_scale_w = _quant_blockwise(
                go_wgrad.reshape([-1, b * sq]), "1x128"
            )
            go_fp8_w = go_fp8_w.reshape([num_groups, o_lora_rank, b * sq])
            go_scale_w = go_scale_w.reshape([num_groups, o_lora_rank, -1])

            x_fp8_w = x_fp8_w.transpose([2, 0, 1]).contiguous()
            x_scale_w = x_scale_w.transpose([2, 0, 1]).contiguous()
            go_fp8_w = go_fp8_w.transpose([2, 0, 1]).contiguous()
            go_scale_w = go_scale_w.transpose([2, 0, 1]).contiguous()

            grad_weight = paddle.empty(
                [num_groups, d, o_lora_rank], dtype=paddle.bfloat16
            )
            deep_gemm.fp8_einsum(
                "bhd,bhr->hdr",
                (x_fp8_w, x_scale_w),
                (go_fp8_w, go_scale_w),
                grad_weight,
                recipe=(1, 1, 128),
            )
            # [g, d, r] -> [g, r, d]
            grad_weight = grad_weight.transpose([0, 2, 1]).contiguous()
        else:
            if x is None:
                x_fp8_2d = ctx.x_fp8_w_pre.reshape([num_groups * d, b * sq])
                x_scale_2d = ctx.x_scale_w_pre.reshape([num_groups * d, -1])
                x_bf16_2d = paddle.incubate.nn.functional.fused_act_dequant(
                    x_fp8_2d, x_scale_2d
                )
                x = (
                    x_bf16_2d.reshape([num_groups, d, b * sq])
                    .transpose([2, 0, 1])
                    .reshape([b, sq, num_groups, d])
                )
            grad_weight = paddle.einsum("bsgd,bsgr->grd", x, grad_output)

        return (
            grad_x if ctx.x_needs_grad else None,
            grad_weight.reshape([num_groups * o_lora_rank, d]),
        )


def _validate_dsv4_boundary_values(
    startend_row_indices: Tensor,
    upper_bound: int,
    description: str,
    *,
    require_per_sample_max: bool = False,
) -> None:
    """Validate exclusive document endpoints against a sequence bound."""
    min_endpoint = int(paddle.min(startend_row_indices).item())
    max_endpoint = int(paddle.max(startend_row_indices).item())
    if min_endpoint < 0 or max_endpoint > upper_bound:
        raise ValueError(
            f"{description} document endpoints must be in [0, {upper_bound}], "
            f"got range [{min_endpoint}, {max_endpoint}]"
        )

    if require_per_sample_max:
        per_sample_max = paddle.max(
            startend_row_indices.reshape([startend_row_indices.shape[0], -1]),
            axis=1,
        ).tolist()
        if any(int(endpoint) != upper_bound for endpoint in per_sample_max):
            raise ValueError(
                f"each sample must end at {upper_bound}, got maximum "
                f"document endpoints {per_sample_max}"
            )


def pack_dsv4_docmask(
    startend_row_indices: Tensor | None,
    batch_size: int,
    seqlen: int,
    *,
    cp_size: int,
    dense_mode: bool,
    max_sequence_length: int | None = None,
) -> Tensor:
    """Rebase a ``[b, 1, s, 1]`` document mask onto the packed ``[1, 1, b*s, 1]``.

    The mask half of ``_pack_dsv4_logical_batch``, guards included, factored out
    because the ``csa_share_docmask_meta`` prebuild has to produce exactly the
    layout the layers will look up. A second copy of the offset arithmetic there
    would be invisible to the registry's consistency check, which compares
    ``(ratio, batch_size, seqlen)`` only: a drifted offset rule keeps every shape
    right and silently serves the wrong document boundaries.

    ``batch_size > 1`` is the caller's precondition -- at 1 there is nothing to
    rebase and the mask is already in its packed form.
    """
    if not dense_mode:
        raise NotImplementedError(
            "DSv4HybridAttention only supports batch_size > 1 in dense mode; "
            "indexer mode is unsupported"
        )
    if cp_size > 1:
        raise NotImplementedError(
            "DSv4HybridAttention does not support batch_size > 1 with "
            f"context parallelism (batch_size={batch_size}, cp_size={cp_size})"
        )

    if startend_row_indices is None:
        raise ValueError(
            "DSv4HybridAttention requires startend_row_indices when "
            "batch_size > 1 to preserve sample isolation"
        )

    expected_shape = [batch_size, 1, seqlen, 1]
    if list(startend_row_indices.shape) != expected_shape:
        raise ValueError(
            "startend_row_indices must have shape "
            f"{expected_shape} for batch_size > 1, got "
            f"{list(startend_row_indices.shape)}"
        )
    if max_sequence_length is not None and int(max_sequence_length) != seqlen:
        raise ValueError(
            "DSv4HybridAttention batch packing requires "
            "max_sequence_length to equal the per-sample sequence length; "
            f"got max_sequence_length={max_sequence_length}, S={seqlen}"
        )
    _validate_dsv4_boundary_values(
        startend_row_indices,
        seqlen,
        "unpacked",
        require_per_sample_max=True,
    )
    sample_offsets = (
        paddle.arange(batch_size, dtype=startend_row_indices.dtype).reshape(
            [batch_size, 1, 1, 1]
        )
        * seqlen
    )
    startend_row_indices = (startend_row_indices + sample_offsets).reshape(
        [1, 1, batch_size * seqlen, 1]
    )
    _validate_dsv4_boundary_values(
        startend_row_indices,
        batch_size * seqlen,
        "packed",
    )
    return startend_row_indices


def _pack_dsv4_logical_batch(
    hidden_states: Tensor,
    startend_row_indices: Tensor | None,
    *,
    cp_size: int,
    dense_mode: bool,
    max_sequence_length: int | None = None,
) -> tuple[Tensor, Tensor | None, int, int]:
    """Pack a logical batch into the single-sequence DSV4 representation."""
    if len(hidden_states.shape) != 3:
        raise ValueError(
            "DSv4HybridAttention expects rank-3 hidden_states [B, S, H], "
            f"got shape {hidden_states.shape}"
        )

    batch_size, seqlen, _ = hidden_states.shape
    if batch_size <= 1:
        return hidden_states, startend_row_indices, batch_size, seqlen

    startend_row_indices = pack_dsv4_docmask(
        startend_row_indices,
        batch_size,
        seqlen,
        cp_size=cp_size,
        dense_mode=dense_mode,
        max_sequence_length=max_sequence_length,
    )

    hidden_states = hidden_states.reshape([1, batch_size * seqlen, -1])
    return hidden_states, startend_row_indices, batch_size, seqlen


def _unpack_dsv4_logical_batch(
    output: Tensor, batch_size: int, seqlen: int
) -> Tensor:
    """Restore DSV4 output to the caller's logical batch shape."""
    if len(output.shape) != 3:
        raise ValueError(
            "DSv4HybridAttention output must be rank 3 before unpacking, "
            f"got shape {output.shape}"
        )
    return output.reshape([batch_size, seqlen, -1])


from paddlefleet.transformer.utils import (
    get_doc_lens,
)


def build_document_rope_freqs(
    rotary_pos_emb: nn.Layer,
    sq: int,
    startend_row_indices: Tensor | None = None,
    position_offset: int = 0,
    doc_lens: Tensor | None = None,
):
    """Build RoPE frequencies that restart from zero for each document.

    Args:
        rotary_pos_emb: the layer's RotaryEmbedding / YarnRotaryEmbedding.
        sq: local query sequence length.
        startend_row_indices: optional ``[1, 1, seqlen, 1]`` document
            boundary tensor. Required only when ``doc_lens`` is not provided.
        position_offset: global position offset for CP (``cp_rank * sq``);
            the returned freqs cover ``[0, position_offset + sq)`` and are
            sliced by the caller.
        doc_lens: optional precomputed document lengths (e.g. from
            ``CSADocMaskMetadata.doc_lens``) to avoid recomputing them from
            ``startend_row_indices``.

    Returns:
        (freqs, mscale): ``freqs`` is ``[1, position_offset + sq, 1, head_dim]``
        and ``mscale`` is the YaRN mscale (DSv4 forces it to 1.0 downstream).
    """
    if doc_lens is None:
        assert startend_row_indices is not None, (
            "Document RoPE requires startend_row_indices when doc_lens is not provided."
        )
        assert (
            startend_row_indices.shape[0] == 1
            and startend_row_indices.shape[1] == 1
        ), "Document RoPE currently expects batch_size == 1 and head == 1."
        doc_lens = get_doc_lens(startend_row_indices)

    max_doc_len = int(doc_lens.max().item())
    _rope_result = rotary_pos_emb(max_doc_len, packed_seq=False)
    if isinstance(_rope_result, tuple):
        freqs, mscale = _rope_result
    else:
        freqs, mscale = _rope_result, 1.0
    freqs = freqs.squeeze(0).squeeze(1)
    doc_freqs = [freqs[:doc_len] for doc_len in doc_lens.tolist()]
    freqs = paddle.concat(doc_freqs, axis=0)
    needed_len = position_offset + sq
    if freqs.shape[0] < needed_len:
        freqs = paddle.concat(
            [
                freqs,
                paddle.zeros(
                    [needed_len - freqs.shape[0], freqs.shape[-1]],
                    dtype=freqs.dtype,
                ),
            ],
            axis=0,
        )

    return freqs.reshape([1, -1, 1, freqs.shape[-1]]), mscale


def _build_rope_freqs(
    rotary_pos_emb: nn.Layer,
    sq: int,
    position_offset: int = 0,
    docmask_meta: CSADocMaskMetadata | None = None,
    startend_row_indices: Tensor | None = None,
):
    if docmask_meta is not None:
        _rope_result = rotary_pos_emb(docmask_meta.seqlen, packed_seq=False)
        if isinstance(_rope_result, tuple):
            freqs, mscale = _rope_result
        else:
            freqs, mscale = _rope_result, 1.0
        freqs = paddle.gather(freqs, docmask_meta.pos_in_doc, axis=1)
    elif startend_row_indices is not None:
        freqs, mscale = build_document_rope_freqs(
            rotary_pos_emb,
            sq,
            startend_row_indices=startend_row_indices,
            position_offset=position_offset,
        )
    else:
        _rope_result = rotary_pos_emb(sq + position_offset, packed_seq=False)
        if isinstance(_rope_result, tuple):
            freqs, mscale = _rope_result
        else:
            freqs, mscale = _rope_result, 1.0
    return freqs[:, position_offset : position_offset + sq, :], mscale


# ---------------------------------------------------------------------------
# Sublayers spec dataclass
# ---------------------------------------------------------------------------


@dataclass
class DSv4HybridSelfAttentionSublayersSpec:
    """Sublayer specifications for DSv4 Hybrid Self-Attention."""

    linear_q_down_proj: type | LayerSpec = None
    linear_q_up_proj: type | LayerSpec = None
    linear_kv_proj: type | LayerSpec = None
    core_attention: type | LayerSpec | None = None
    o_proj: type | LayerSpec = None
    q_layernorm: type | LayerSpec = None
    kv_layernorm: type | LayerSpec = None
    gate_proj: type | LayerSpec | None = None


# ---------------------------------------------------------------------------
# DSv4HybridAttention
# ---------------------------------------------------------------------------


class DSv4HybridAttention(Attention):
    """DSv4 Hybrid Attention with CSA core attention, inverse RoPE, and grouped output.

    This class:
    1. Builds per-layer RotaryEmbedding (with configurable base for compressed layers)
    2. Builds CompressedSparseAttention as core attention
    3. Applies inverse RoPE on attention output
    4. Performs grouped low-rank output projection
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: DSv4HybridSelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.num_attention_heads = config.num_attention_heads
        self.v_head_dim = config.v_head_dim
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim or 0
        self.query_projection_size = self.num_attention_heads * self.v_head_dim
        self.q_head_dim = self.v_head_dim
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.v_head_dim

        # Per-layer compress ratio
        if is_mtp_layer:
            layer_idx = self.config.num_hidden_layers + layer_number
            compress_ratio = self.config.csa_compress_ratios[layer_idx]
        else:
            layer_idx = layer_number - self.config.num_empty_layers_add_in_head
            compress_ratio = self.config.csa_compress_ratios[layer_idx]
        assert compress_ratio != -2, (
            "DSv4HybridAttention should not be constructed for MLA ratio -2"
        )
        if compress_ratio not in {-1, 0, 128} and not 2 <= compress_ratio < 128:
            raise ValueError(
                f"DSv4 hybrid attention requires HCA/CSA/window ratio, got {compress_ratio}"
            )

        # Which logical document mask this layer reads, used as part of the
        # shared-metadata cache key (config.csa_share_docmask_meta). Main layers
        # all read the same decoder mask and therefore share a group; every MTP
        # depth is fed its own slice of ``mtp_startend_row_indices_all``, which
        # has the same shape as the main mask but different contents, so each
        # gets its own group. Deliberately independent of compress_ratio.
        self.csa_mask_group = (
            ("mtp", int(layer_number)) if is_mtp_layer else ("main",)
        )

        # Resolve the per-attention-type RoPE variant override.
        # HCA layers: compress_ratio == 128; CSA layers: 2 <= compress_ratio < 128.
        # When the per-type field is unset (None), the historical default below
        # is kept so existing configs behave exactly as before.
        if compress_ratio == 128:
            per_type_rope_type = config.hca_rope_type
        elif 2 <= compress_ratio < 128:
            per_type_rope_type = config.csa_rope_type
        else:
            per_type_rope_type = None

        # Per-layer RoPE (potentially different base for compressed layers)
        rope_base = getattr(config, "rotary_base", 10000)
        if compress_ratio > 1:
            # Every shipped model_config.json writes csa_compress_rotary_base as
            # a *string* ("160000.0"). pretrain.py coerces numeric strings
            # before building the config, but any path that calls from_config
            # directly (unit tests, offline inference, tooling) would reach
            # YarnRotaryEmbedding's math.log() with a str and raise TypeError.
            rope_base = float(config.csa_compress_rotary_base)

        # Resolve the RoPE variant for this layer. Historically compressed
        # layers (compress_ratio > 1, i.e. HCA/CSA) always used YaRN while
        # window/MQA layers used plain RoPE. The per-attention-type override
        # (hca_rope_type / csa_rope_type) lets HCA and CSA independently pick
        # "rope" or "yarn"; when unset the historical default is preserved.
        default_rope_type = "yarn" if compress_ratio > 1 else "rope"
        resolved_rope_type = per_type_rope_type or default_rope_type

        if resolved_rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=rope_base,
                scaling_factor=getattr(config, "rotary_scaling_factor", 40),
                original_max_position_embeddings=getattr(
                    config, "original_max_position_embeddings", 4096
                ),
                beta_fast=getattr(config, "beta_fast", 32),
                beta_slow=getattr(config, "beta_slow", 1),
                mscale=getattr(config, "mscale", 1.0),
                mscale_all_dim=getattr(config, "mscale_all_dim", 0.0),
                yarn_rope_fusion=getattr(
                    config, "dsv4_yarn_rope_fusion", False
                ),
            )
        else:
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=getattr(config, "rotary_percent", 1.0),
                rotary_base=rope_base,
            )

        self.core_attention = build_spec_layer(
            sublayers_spec.core_attention,
            config=config,
            layer_number=layer_number if is_mtp_layer else layer_idx + 1,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=None,
            softmax_scale=getattr(config, "softmax_scale", None),
            k_channels=self.q_head_dim,
            v_channels=self.v_head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=1,
            cp_comm_type=cp_comm_type,
            pg_collection=self.pg_collection,
            is_mtp_layer=is_mtp_layer,
            compress_ratio=compress_ratio,
            rotary_pos_emb=self.rotary_pos_emb,
        )

        # Grouped output projection
        self.o_local_groups = config.o_groups
        assert self.query_projection_size % config.o_groups == 0, (
            "num_attention_heads * v_head_dim must be divisible by o_groups"
        )
        group_proj_in_size = self.query_projection_size // config.o_groups
        group_proj_out_size = config.o_groups * config.o_lora_rank

        self.linear_o_group_proj = self.create_parameter(
            shape=[group_proj_out_size, group_proj_in_size],
            dtype=config.dtype if hasattr(config, "dtype") else "bfloat16",
            default_initializer=nn.initializer.Normal(
                std=getattr(config, "init_method_std", 0.02)
            ),
        )

        linear_proj_in_size = config.o_groups * config.o_lora_rank
        self.use_fp8_qat = getattr(config, "use_fp8_qat", False)
        self.o_proj = build_spec_layer(
            sublayers_spec.o_proj,
            linear_proj_in_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        # Gated attention. For MQA the gate multiplies the output of the grouped
        # low-rank projection (linear_o_group_proj), right before o_proj.
        self.gated_attention = getattr(config, "gated_attention", False)
        self.gated_attn_use_q_lora = getattr(
            config, "gated_attn_use_q_lora", False
        )
        if self.gated_attention and sublayers_spec.gate_proj is not None:
            # Gate input source: q_compressed (post q_layernorm, dim=q_lora_rank)
            # when gated_attn_use_q_lora is set, otherwise the full hidden_states.
            if self.gated_attn_use_q_lora:
                assert config.q_lora_rank is not None, (
                    "gated_attn_use_q_lora=True requires q_lora_rank is not None"
                )
                gate_in_dim = config.q_lora_rank
            else:
                gate_in_dim = config.hidden_size
            self.gate_proj = build_spec_layer(
                sublayers_spec.gate_proj,
                gate_in_dim,
                linear_proj_in_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=self.config.use_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_group=self.pg_collection.tp,
            )
        else:
            self.gated_attention = False
            self.gate_proj = None

        self.recompute_gated_attn = (
            config.recompute_granularity == "selective"
            and module_needs_recompute("gated_attn", self.layer_number, config)
        )

        self.recompute_full_attn = (
            config.recompute_granularity == "selective"
            and module_needs_recompute("full_attn", self.layer_number, config)
        )
        self._full_attn_recompute = None
        self._gate_recompute = None

        # VHA postmix: low-rank cross-head mixing of the attention output, applied
        # after inverse RoPE (head space) and before the grouped output projection.
        # Two topologies, selected by config.vha_postmix_grouped:
        #   grouped=False (default): ungrouped full cross-head mixing over all heads
        #                  (the earlier VHA design). Higher capacity.
        #   grouped=True:  within-group block-diagonal mixing (per o_group); mixing
        #                  stays within each o_group.
        # Reuses the shared use_vha_attention / vha_postmix_rank knobs; premix is
        # wired separately (use_vha_premix).
        self.use_vha_postmix = getattr(config, "use_vha_attention", False)
        self.vha_postmix_grouped = getattr(config, "vha_postmix_grouped", False)
        if self.use_vha_postmix:
            group_heads = self.num_attention_heads // self.o_local_groups
            if self.vha_postmix_grouped:
                # Grouped postmix mixes heads within each o_group on the
                # group_heads axis, so it requires an exact head split across
                # groups. The constructor only guarantees
                # (num_attention_heads * v_head_dim) % o_groups == 0, which is
                # weaker (e.g. nh=6, v_head_dim=4, o_groups=4 passes but yields
                # group_heads=1 and a 6->4 head reshape mismatch). Enforce the
                # stronger head-level divisibility here with an explicit
                # ValueError (assertions are stripped under `python -O`).
                if self.num_attention_heads % self.o_local_groups != 0:
                    raise ValueError(
                        "grouped VHA postmix requires num_attention_heads "
                        f"({self.num_attention_heads}) to be divisible by "
                        f"o_groups ({self.o_local_groups})"
                    )
                # Per-group mixing on the group_heads axis; rank capped at
                # group_heads (full-rank), beyond which it is redundant.
                mix_heads = group_heads
                u_shape = [self.o_local_groups, group_heads, None]
            else:
                # Full cross-head mixing on the nh axis; rank capped at nh.
                mix_heads = self.num_attention_heads
                u_shape = [self.num_attention_heads, None]
            vha_postmix_rank = config.vha_postmix_rank
            if vha_postmix_rank is None:
                vha_postmix_rank = mix_heads // 4
            vha_postmix_rank = max(1, min(vha_postmix_rank, mix_heads))
            self.vha_postmix_rank = vha_postmix_rank
            u_shape[-1] = vha_postmix_rank
            self.vha_postmix_U = self.create_parameter(
                shape=u_shape,
                default_initializer=nn.initializer.Normal(mean=0.0, std=0.01),
            )
            self.vha_postmix_V = self.create_parameter(
                shape=u_shape,
                default_initializer=nn.initializer.Constant(0.0),
            )
        # Selective recompute for the VHA postmix scatter chain (independently
        # configurable via the "vha_postmix" entry in recompute_modules),
        # configured like every other selective submodule.
        self.recompute_vha_postmix = (
            config.recompute_granularity == "selective"
            and module_needs_recompute("vha_postmix", self.layer_number, config)
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor, None]:
        """Forward pass.

        Args:
            hidden_states: [b, sq, hidden_size]
            attention_mask: optional mask

        Returns:
            (output [b, sq, hidden_size], bias=None)
        """
        startend_row_indices = kwargs.get(
            "attn_mask_startend_row_indices", None
        )

        # KV cache (incremental decode): duck-typed CSADynamicCache.
        past_key_values = kwargs.get("past_key_values", None)
        layer_idx = kwargs.get("layer_idx", None)
        use_cache = kwargs.get("use_cache", False)

        # Get Q, K, V tensors
        # In CP mode, pass position_offset so RoPE uses correct global positions.
        cp_pg = getattr(self, "pg_collection", None)
        cp_pg = cp_pg.cp if cp_pg is not None else None
        cp_size = getattr(cp_pg, "nranks", 1) if cp_pg is not None else 1
        if cp_size > 1:
            assert self.config.cp_balance_mode == "contiguous_allgather", (
                f"DSv4HybridAttention requires cp_balance_mode='contiguous_allgather', "
                f"got '{self.config.cp_balance_mode}'"
            )
        cp_rank = (
            getattr(cp_pg, "rank", 0)
            if cp_pg is not None and cp_size > 1
            else 0
        )
        hidden_states, startend_row_indices, original_b, original_sq = (
            _pack_dsv4_logical_batch(
                hidden_states,
                startend_row_indices,
                cp_size=cp_size,
                dense_mode=self.config.csa_dense_mode,
                max_sequence_length=getattr(
                    self.config, "max_sequence_length", None
                ),
            )
        )
        b, sq, _ = hidden_states.shape
        position_offset = cp_rank * sq if cp_size > 1 else 0

        # Incremental decode: the new token's absolute position is the number
        # of raw tokens already cached; RoPE (forward + inverse) must use it.
        if use_cache and past_key_values is not None and cp_size == 1:
            position_offset = past_key_values.get_csa_state(
                layer_idx
            ).raw_seq_len()

        docmask_meta = None
        ratio = int(getattr(self.core_attention, "compress_ratio", 0))
        if startend_row_indices is not None:
            docmask_seqlen = sq * cp_size if cp_size > 1 else sq
            # csa_share_docmask_meta: the trainer built this step's metadata
            # before the forward started. The micro-batch slot was picked in
            # TransformerLayer.forward, i.e. outside any recompute wrapper, and
            # handed down as docmask_mb_idx; the lookup is a pure read, so it is safe
            # here even when this forward is being replayed by recompute.
            docmask_mb_idx = kwargs.get("docmask_mb_idx", -1)
            if (
                getattr(self.config, "csa_share_docmask_meta", False)
                and docmask_mb_idx >= 0
            ):
                from paddlefleet.transformer.doc_mask_meta_registry import (
                    doc_mask_meta_registry,
                )

                # The trainer prebuilds only the main group: the MTP depths are
                # fed a slice of their own mtp_startend_row_indices_all, absent
                # from the store by design, so the lookup returns None there and
                # the layer builds its own metadata below, mirroring
                # MQALatentAttention. A main-group miss keeps raising.
                docmask_meta = doc_mask_meta_registry.get(
                    docmask_mb_idx,
                    max(1, ratio),
                    b,
                    docmask_seqlen,
                    self.csa_mask_group,
                )
            if docmask_meta is None:
                docmask_meta = CSADocMaskMetadata.build(
                    max(1, ratio),
                    b,
                    docmask_seqlen,
                    startend_row_indices,
                    dense_mode=self.config.csa_dense_mode,
                )

        # Full attention recompute: wrap qkv + core_attn + inv_rope + o_group_proj + gated_attn
        # into one RecomputeWithoutOutput block. All intermediates (query, key, value, etc.)
        # are freed after forward (no_grad), and the output is discarded after o_proj.
        input_ids = kwargs.get("input_ids", None)
        if self.recompute_full_attn and self.training:
            self._full_attn_recompute = RecomputeWithoutOutput()
            core_attn_out = self._full_attn_recompute.recompute(
                self._full_attn_forward,
                # This segment contains core_attention, i.e. the CSA Indexer and its
                # side-attached loss. RecomputeWithoutOutput is a PyLayer whose
                # output is differentiable only if some input is, and with a frozen
                # backbone (train_indexer_only) hidden_states is detached. It
                # would then skip registering its recompute hook altogether
                # (tensor_parallel/random.py:590 checks stop_gradient) and the
                # Indexer would get no gradient, with no error and no warning.
                keep_indexer_grad_path(hidden_states, self.config),
                attention_mask,
                position_offset,
                docmask_meta,
                input_ids,
                True,  # _in_full_recompute (last positional; see signature)
                preserve_rng_state=False,
                share_grad_holder=True,
            )

            # Output projection
            output, bias = deferrable_linear(
                self.config, "attn_out_proj", self.o_proj, core_attn_out
            )

            # Discard full_attn output (core_attn_out) — frees ~512 MB
            self._full_attn_recompute.discard_output_and_register_recompute(
                output
            )
            self._full_attn_recompute = None
        else:
            core_attn_out = self._full_attn_forward(
                hidden_states,
                attention_mask,
                position_offset,
                docmask_meta,
                input_ids,
                past_key_values=past_key_values,
                layer_idx=layer_idx,
                use_cache=use_cache,
            )

            # Output projection
            output, bias = deferrable_linear(
                self.config, "attn_out_proj", self.o_proj, core_attn_out
            )

            # Discard gated_attn output if it was independently recomputed
            if (
                hasattr(self, "_gate_recompute")
                and self._gate_recompute is not None
            ):
                self._gate_recompute.discard_output_and_register_recompute(
                    output
                )
                self._gate_recompute = None

        if original_b > 1:
            output = _unpack_dsv4_logical_batch(output, original_b, original_sq)

        return output, bias

    def _full_attn_forward(
        self,
        hidden_states: Tensor,
        attention_mask,
        position_offset: int,
        docmask_meta,
        input_ids,
        _in_full_recompute: bool = False,
        *,
        past_key_values=None,
        layer_idx=None,
        use_cache=False,
    ) -> Tensor:
        """Full attention forward: qkv_proj + core_attn + inv_rope + o_group_proj + gated_attn.

        Factored out so that paddle.distributed.fleet.utils.recompute can wrap it.
        The large intermediate tensors (query, key, value) are internal to this function
        and will be freed after this function returns during forward, then recomputed
        during backward.

        ``RecomputeWithoutOutput.recompute()`` forwards only ``*args`` to the
        wrapped function, so ``_in_full_recompute`` must stay the *last*
        positional parameter. The KV-cache parameters after it are keyword-only
        to keep that invariant: they are never used from the recompute path
        (recompute only runs under ``self.training``).
        """
        query, key, value, q_compressed, kv_compressed = (
            self.get_query_key_value_tensors(
                hidden_states=hidden_states,
                position_offset=position_offset,
                docmask_meta=docmask_meta,
            )
        )

        # Core attention (CompressedSparseAttention)
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            x=hidden_states,
            qr=q_compressed,
            input_ids=input_ids,
            docmask_meta=docmask_meta,
            past_key_values=past_key_values,
            layer_idx=layer_idx,
            use_cache=use_cache,
        )
        # core_attn_out: [b, sq, np * v_head_dim]

        # Inverse RoPE on last qk_pos_emb_head_dim of each head
        b, sq, _ = core_attn_out.shape
        pos_dim = self.qk_pos_emb_head_dim
        nope_dim = self.v_head_dim - pos_dim
        postmix_done = False

        if pos_dim > 0:
            core_attn_out = core_attn_out.reshape(
                [b, sq, self.num_attention_heads, self.v_head_dim]
            )
            freqs, mscale = _build_rope_freqs(
                self.rotary_pos_emb,
                sq,
                position_offset=position_offset,
                docmask_meta=docmask_meta,
            )
            # DSv4 reference uses pure norm-preserving RoPE; YaRN's mscale is not applied.
            mscale = 1.0

            if self._can_fuse_inv_rope_postmix(_in_full_recompute):
                # Fused inverse RoPE + ungrouped VHA postmix. Bitwise identical
                # to running the two separately, but never materialises the
                # full-width rotated output. It consumes the postmix, so the
                # standalone postmix block below must be skipped.
                core_attn_out = self._apply_inv_rope_vha_postmix(
                    core_attn_out, freqs, nope_dim, pos_dim, mscale
                )
                postmix_done = True
            elif (
                self.config.apply_rope_fusion
                and not self.config.high_precision_rope
            ):
                from paddlefleet.triton_ops import fused_apply_mla_rope_inplace

                core_attn_out = fused_apply_mla_rope_inplace(
                    core_attn_out,
                    freqs,
                    nope_dim,
                    mscale,
                    inverse=True,
                    clone_input=True,
                )
            else:
                content_part = core_attn_out[..., :nope_dim]
                rot_part = core_attn_out[..., nope_dim:]

                rot_part = _apply_rotary_pos_emb_bshd(
                    rot_part,
                    freqs,
                    mscale=mscale,
                    rotary_interleaved=False,
                    multi_latent_attention=True,
                    inverse=True,
                    mla_output_remove_interleaving=True,
                    high_precision_rope=self.config.high_precision_rope,
                )
                core_attn_out = paddle.concat([content_part, rot_part], axis=-1)
                core_attn_out = core_attn_out.reshape([b, sq, -1])

        # VHA postmix: low-rank cross-head mixing while still in head space
        # ([b, sq, nh, v_head_dim]), after inverse RoPE and before the grouped
        # output projection. When the whole block is already wrapped in a
        # full_attn RecomputeWithoutOutput, skip the nested selective recompute
        # (the full block recompute already frees these activations).
        if self.use_vha_postmix and not postmix_done:
            if (
                self.recompute_vha_postmix
                and self.training
                and not _in_full_recompute
            ):
                core_attn_out = recompute(
                    self._apply_vha_postmix, core_attn_out
                )
            else:
                core_attn_out = self._apply_vha_postmix(core_attn_out)

        # Grouped output projection
        core_attn_out = core_attn_out.reshape([b, sq, self.o_local_groups, -1])
        if (
            self.config.fp8 is not None
            and self.config.full_fp8_computation
            and FLEET_FP8_WO_A_GEMM
        ):
            core_attn_out = GroupedOutputFP8.apply(
                core_attn_out,
                self.linear_o_group_proj,
                self.o_local_groups,
                self.config.o_lora_rank,
                self.config.fp8_wgrad,
            )
        else:
            group_shape = [
                self.o_local_groups,
                self.config.o_lora_rank,
                self.linear_o_group_proj.shape[-1],
            ]
            from paddlefleet.triton_ops import fused_grouped_matmul

            dw_acc = deferred_grouped_dw_accumulator(
                self.config, "attn_o_group_proj", self.linear_o_group_proj
            )
            if dw_acc is None:
                core_attn_out = fused_grouped_matmul(
                    core_attn_out, self.linear_o_group_proj.reshape(group_shape)
                )
            else:
                # Hand in the leaf parameter, not a reshaped view: the deferred path
                # returns None for the weight grad, which would cut the chain back
                # to the parameter if the input were a non-leaf view.
                core_attn_out = fused_grouped_matmul(
                    core_attn_out,
                    self.linear_o_group_proj,
                    dw_accumulator=dw_acc,
                    group_shape=group_shape,
                )
            core_attn_out = core_attn_out.reshape([b, sq, -1])

        # Apply gated attention
        if self.gated_attention:
            gate_source = (
                q_compressed if self.gated_attn_use_q_lora else hidden_states
            )
            # When NOT inside full_attn recompute, gated_attn can have its own
            # independent RecomputeWithoutOutput wrapper for lighter memory saving.
            if (
                self.recompute_gated_attn
                and self.training
                and not _in_full_recompute
            ):
                self._gate_recompute = RecomputeWithoutOutput()
                core_attn_out = self._gate_recompute.recompute(
                    self._gate,
                    gate_source,
                    core_attn_out,
                    preserve_rng_state=False,
                    share_grad_holder=True,
                )
            else:
                core_attn_out = self._gate(gate_source, core_attn_out)

        return core_attn_out

    def _gate(self, gate_source: Tensor, core_attn_out: Tensor) -> Tensor:
        gate, _ = deferrable_linear(
            self.config, "attn_gate_proj", self.gate_proj, gate_source
        )
        if getattr(self.config, "sigmoid_gate_fusion", False):
            from paddlefleet.triton_ops import SigmoidGateFusionTriton

            core_attn_out = SigmoidGateFusionTriton.apply(core_attn_out, gate)
        else:
            core_attn_out = core_attn_out * paddle.nn.functional.sigmoid(gate)
        return core_attn_out

    def _can_fuse_inv_rope_postmix(self, in_full_recompute: bool) -> bool:
        """Whether the inverse RoPE can be folded into the postmix GEMM.

        Every rejected case falls back to the unfused pair, so this is a pure
        performance switch: the eager RoPE path, high_precision_rope, the
        grouped postmix topology (einsum, no [nh,nh] GEMM to split) and the
        postmix's own selective recompute wrapper all keep working unchanged.
        """
        if not getattr(self.config, "fuse_inv_rope_into_vha_postmix", False):
            return False
        if not self.use_vha_postmix or self.vha_postmix_grouped:
            return False
        if not self.config.apply_rope_fusion:
            return False
        if self.config.high_precision_rope:
            return False
        # Re-entering the fused PyLayer from a nested recompute wrapper buys
        # nothing (the fusion already avoids the intermediate it would free).
        if (
            self.recompute_vha_postmix
            and self.training
            and not in_full_recompute
        ):
            return False
        return True

    def _apply_inv_rope_vha_postmix(
        self,
        attn_out: Tensor,
        freqs: Tensor,
        nope_dim: int,
        pe_dim: int,
        mscale: float,
    ) -> Tensor:
        """Inverse RoPE + ungrouped VHA postmix in one pass.

        attn_out: [b, sq, nh, v_head_dim]. Returns [b, sq, nh * v_head_dim],
        matching what the unfused RoPE followed by ``_apply_vha_postmix`` would
        return, bit for bit. The postmix matrix is rebuilt inside the fused op
        exactly as ``_apply_vha_postmix``'s ungrouped branch builds it.
        """
        from paddlefleet.triton_ops import fused_inv_rope_vha_postmix

        return fused_inv_rope_vha_postmix(
            attn_out,
            freqs,
            self.vha_postmix_U,
            self.vha_postmix_V,
            nope_dim,
            pe_dim,
            mscale,
        )

    def _apply_vha_postmix(self, attn_out: Tensor) -> Tensor:
        """Low-rank cross-head mixing of the attention output.

        attn_out: [b, sq, nh * v_head_dim] (head space, post inverse RoPE).

        Two topologies (config.vha_postmix_grouped):

        - grouped=True: reshape to [b, sq, o_groups, group_heads, v_head_dim] and,
          within each o_group, recombine that group's heads via a low-rank
          correction (I + U_g V_g^T) on the head axis, shared across v_head_dim.
          Block-diagonal w.r.t. o_groups (mixing stays within a group).

        - grouped=False: reshape to [b, sq, nh, v_head_dim] and recombine ALL heads
          via a single low-rank correction (I + U V^T) on the nh axis. Full
          cross-head mixing (higher capacity).

        V is zero-initialized so this is identity at the start of training.
        """
        b, sq = attn_out.shape[0], attn_out.shape[1]
        if self.vha_postmix_grouped:
            g = self.o_local_groups
            gh = self.num_attention_heads // g
            mixed = attn_out.reshape([b, sq, g, gh, self.v_head_dim])
            z = paddle.einsum("btgjd,gjr->btgrd", mixed, self.vha_postmix_U)
            delta = paddle.einsum("btgrd,gjr->btgjd", z, self.vha_postmix_V)
            mixed = mixed + delta
            return mixed.reshape(
                [b, sq, self.num_attention_heads * self.v_head_dim]
            )
        # ungrouped: fused dense M = I + V @ U^T, then a single [nh,nh] GEMM on
        # the head axis. Rank-independent and faster than the split low-rank
        # form (whose r-sized contraction underutilizes the GPU); differs from
        # the two-matmul path only by bf16 contraction order.
        nh, d = self.num_attention_heads, self.v_head_dim
        mixed = attn_out.reshape([b * sq, nh, d])
        M = paddle.matmul(
            self.vha_postmix_V, self.vha_postmix_U, transpose_y=True
        )  # [nh,r]@[r,nh]->[nh,nh]
        M = M + paddle.eye(nh, dtype=M.dtype)
        out = paddle.matmul(M, mixed)  # [nh,nh]@[B,nh,d]->[B,nh,d]
        return out.reshape([b, sq, nh * d])

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        startend_row_indices: Tensor | None = None,
        position_offset: int = 0,
        docmask_meta: CSADocMaskMetadata | None = None,
    ):
        """Override in subclass."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DSv4HybridSelfAttention
# ---------------------------------------------------------------------------


class DSv4HybridSelfAttention(DSv4HybridAttention):
    """DSv4 Hybrid Self-Attention with Q low-rank decomposition and single-head KV.

    Q path: hidden -> q_down_proj -> q_layernorm -> q_up_proj -> rms_norm -> RoPE
    KV path: hidden -> kv_proj -> kv_layernorm -> RoPE (single head, key == value)
    """

    def __init__(
        self,
        config: TransformerConfig,
        sublayers_spec: DSv4HybridSelfAttentionSublayersSpec,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(
            config=config,
            sublayers_spec=sublayers_spec,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )

        self.q_lora_rank = config.q_lora_rank
        q_head_dim = self.v_head_dim  # In DSv4 Hybrid, q_head_dim == v_head_dim

        # Q down projection: hidden_size -> q_lora_rank (duplicated)
        self.linear_q_down_proj = build_spec_layer(
            sublayers_spec.linear_q_down_proj,
            config.hidden_size,
            self.q_lora_rank,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            skip_weight_param_allocation=False,
            tp_group=None,
        )

        # Q layernorm
        self.q_layernorm = build_spec_layer(
            sublayers_spec.q_layernorm,
            config=config,
            hidden_size=self.q_lora_rank,
            eps=getattr(config, "rms_norm_eps", 1e-5),
        )

        # Q up projection: q_lora_rank -> num_heads * q_head_dim (column parallel).
        # When use_vha_premix is enabled, linear_q_up_proj is replaced by a
        # structured VHA premix (block-diagonal per group), built below.
        self.use_vha_premix = getattr(
            config, "use_vha_attention", False
        ) and getattr(config, "use_vha_premix", False)
        if self.use_vha_premix:
            g_q = config.vha_premix_groups
            # Explicit ValueError (not assert): assertions are stripped under
            # `python -O`, which would let an invalid grouping silently produce
            # a wrong Q expansion instead of failing fast at construction.
            if g_q is None:
                raise ValueError(
                    "use_vha_premix=True requires config.vha_premix_groups "
                    "to be set"
                )
            if g_q <= 0:
                raise ValueError(
                    f"vha_premix_groups must be a positive integer, got {g_q}"
                )
            if self.num_attention_heads % g_q != 0:
                raise ValueError(
                    f"num_attention_heads ({self.num_attention_heads}) must be "
                    f"divisible by vha_premix_groups ({g_q})"
                )
            if self.q_lora_rank % g_q != 0:
                raise ValueError(
                    f"q_lora_rank ({self.q_lora_rank}) must be divisible by "
                    f"vha_premix_groups ({g_q})"
                )
            self.vha_premix_groups = g_q
            self.vha_premix_expand = self.num_attention_heads // g_q  # k
            self.vha_premix_dq = self.q_lora_rank // g_q  # d_q
            # Structured up-projection (VHA premix, Variant A / shared): the
            # compressed Q is split into g_q groups of d_q dims, and a SINGLE set
            # of k = nh // g_q expansion matrices [d_q, q_head_dim] is broadcast
            # across all groups (weight has no group axis -> 3-D, matching the
            # attention.py premix convention). Head (kk, g) reads only group g's
            # d_q dims, so it is still block-diagonal (mathematically equivalent
            # to a dense up_proj on the compressed Q).
            # Init matches the prior VHA premix (attention.py) non-square branch:
            # each [d_q, q_head_dim] block is semi-orthogonal, scaled by
            # sqrt(q_head_dim / d_q) so the pre-(q_rms_norm) per-element RMS is
            # preserved across the d_q -> q_head_dim expansion. Cannot zero-init
            # (there is no residual around Q).
            premix_scale = math.sqrt(q_head_dim / self.vha_premix_dq)
            init_blocks = []
            for _ in range(self.vha_premix_expand):
                block = paddle.empty([self.vha_premix_dq, q_head_dim])
                nn.initializer.Orthogonal()(block)
                init_blocks.append(block * premix_scale)
            init_weight = paddle.stack(init_blocks).reshape(
                [self.vha_premix_expand, self.vha_premix_dq, q_head_dim]
            )
            self.vha_premix_weight = self.create_parameter(
                shape=[
                    self.vha_premix_expand,
                    self.vha_premix_dq,
                    q_head_dim,
                ],
                default_initializer=nn.initializer.Assign(init_weight),
            )
            self.linear_q_up_proj = None
        else:
            self.linear_q_up_proj = build_spec_layer(
                sublayers_spec.linear_q_up_proj,
                self.q_lora_rank,
                self.num_attention_heads * q_head_dim,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_group=self.pg_collection.tp,
            )
            self.linear_q_up_proj.save_original_input = True

        # KV projection: hidden_size -> v_head_dim (single head)
        self.linear_kv_proj = build_spec_layer(
            sublayers_spec.linear_kv_proj,
            config.hidden_size,
            self.v_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )
        self.linear_kv_proj.save_original_input = True

        # KV layernorm
        self.kv_layernorm = build_spec_layer(
            sublayers_spec.kv_layernorm,
            config=config,
            hidden_size=self.v_head_dim,
            eps=getattr(config, "rms_norm_eps", 1e-5),
        )

    def muon_slice_specs(self, muon_configs):
        """Muon orthogonal-slice specs for the DSv4 hybrid projections."""
        from paddlefleet.transformer.muon_utils import (
            ortho_per_head,
            ortho_stacked,
        )

        if (
            muon_configs.get("muon_qkv_update_mode", "split_head")
            != "split_head"
        ):
            return {}

        specs = {
            # Stored as [o_groups * o_lora_rank, d] but used as [g, r, d] in a
            # grouped gemm, so the leading axis packs o_groups independent
            # matrices and must be split along axis 0.
            "linear_o_group_proj": (
                ortho_per_head,
                {"heads": self.o_local_groups, "axis": -2},
            ),
        }
        if self.use_vha_premix:
            # VHA premix replaces linear_q_up_proj (which is None here). The
            # premix weight is a 3D grouped fuse [k, d_q, q_head_dim] whose
            # leading axis enumerates the k independent expansion matrices, so
            # orthogonalise per leading-axis slice (split along dim 0). Postmix
            # (vha_postmix_U/V) is a plain 2D matrix and is intentionally left
            # unmarked so muon handles it directly.
            specs["vha_premix_weight"] = (ortho_stacked, {})
        else:
            specs["linear_q_up_proj.weight"] = (
                ortho_per_head,
                {"heads": self.num_attention_heads_per_partition},
            )
        if getattr(self, "gate_proj", None) is not None:
            # The gate multiplies the flattened grouped-projection output, whose
            # columns are group-major (group g owns o_lora_rank columns), so the
            # gate weight is fused per o-group rather than per head.
            specs["gate_proj.weight"] = (
                ortho_per_head,
                {"heads": self.o_local_groups},
            )
        return specs

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        startend_row_indices: Tensor | None = None,
        position_offset: int = 0,
        docmask_meta: CSADocMaskMetadata | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Derive query, key, value from hidden_states.

        Args:
            hidden_states: [b, sq, hidden_size]
            startend_row_indices: document boundary tensor, or None.
            position_offset: global position offset for CP (cp_rank * sq_local).
                When non-zero, RoPE frequencies are sliced from the correct
                global starting position.
            docmask_meta: optional :class:`CSADocMaskMetadata` carrying
                precomputed ``doc_lens`` so document RoPE frequencies can be
                built without rescanning ``startend_row_indices``.

        Returns:
            query: [b, sq, num_heads, v_head_dim]
            key:   [b, sq, 1, v_head_dim]
            value: [b, sq, 1, v_head_dim]
            q_compressed: [b, sq, q_lora_rank]
            kv_compressed: [b, sq, hidden_size] (== hidden_states)
        """
        b, sq, _ = hidden_states.shape

        # Q path
        q_compressed, _ = deferrable_linear(
            self.config, "attn_q_proj", self.linear_q_down_proj, hidden_states
        )  # [b, sq, q_lora_rank]
        q_compressed = self.q_layernorm(q_compressed)

        if self.use_vha_premix:
            # Structured premix (Variant A / shared): reshape compressed Q into
            # g_q groups, then expand each group into k heads with a single set of
            # k weight matrices broadcast across all groups. Head layout is
            # kk * g_q + g (k outer, g inner).
            g_q = self.vha_premix_groups
            q = q_compressed.reshape([b, sq, g_q, self.vha_premix_dq])
            q = paddle.einsum(
                "btgr,krd->btkgd", q, self.vha_premix_weight
            )  # [b, sq, k, g_q, q_head_dim]
            q = q.reshape([b, sq, self.num_attention_heads, self.v_head_dim])
        else:
            q, _ = deferrable_linear(
                self.config, "attn_q_proj", self.linear_q_up_proj, q_compressed
            )  # [b, sq, n * v_head_dim]
            q = q.reshape([b, sq, self.num_attention_heads, self.v_head_dim])
        q = _q_rms_norm(
            q,
            getattr(self.config, "rms_norm_eps", 1e-5),
            high_precision_norm=self.config.swa_high_precision_norm,
            use_fusion=getattr(self.config, "dsv4_q_rms_norm_fusion", False),
        )

        # KV path
        kv, _ = deferrable_linear(
            self.config, "attn_kv_proj", self.linear_kv_proj, hidden_states
        )  # [b, sq, v_head_dim]

        if self.config.swa_high_precision_norm:
            kv = self.kv_layernorm(
                kv,
                high_precision_norm=True,
                return_high_precision_norm=True,
            )
        else:
            kv = self.kv_layernorm(kv)

        # Apply RoPE to both Q and KV
        pos_dim = self.qk_pos_emb_head_dim
        nope_dim = self.v_head_dim - pos_dim

        if pos_dim > 0:
            freqs, mscale = _build_rope_freqs(
                self.rotary_pos_emb,
                sq,
                position_offset=position_offset,
                docmask_meta=docmask_meta,
                startend_row_indices=startend_row_indices,
            )
            # DSv4 reference uses pure norm-preserving RoPE; YaRN's mscale is not applied.
            mscale = 1.0

            # Q RoPE: split nope/pe, apply RoPE to pe part
            if (
                self.config.apply_rope_fusion
                and not self.config.high_precision_rope
            ):
                from paddlefleet.triton_ops import fused_apply_mla_rope_inplace

                query = fused_apply_mla_rope_inplace(q, freqs, nope_dim, mscale)
            else:
                q_nope = q[..., :nope_dim]
                q_pe = q[..., nope_dim:]
                q_pe = _apply_rotary_pos_emb_bshd(
                    q_pe,
                    freqs,
                    mscale=mscale,
                    rotary_interleaved=False,
                    multi_latent_attention=True,
                    mla_output_remove_interleaving=True,
                    high_precision_rope=self.config.high_precision_rope,
                )
                query = paddle.concat([q_nope, q_pe], axis=-1)

            # KV RoPE: split nope/pe, apply RoPE to pe part
            if (
                self.config.apply_rope_fusion
                and not self.config.high_precision_rope
                and not self.use_fp8_qat
            ):
                kv = kv.unsqueeze(2)
                kv = fused_apply_mla_rope_inplace(kv, freqs, nope_dim, mscale)
                kv = kv.squeeze(2)
            else:
                kv_nope = kv[..., :nope_dim]
                kv_pe = kv[..., nope_dim:]
                # Add head dim for RoPE: [b, sq, pos_dim] -> [b, sq, 1, pos_dim]
                kv_pe = kv_pe.unsqueeze(2)
                kv_pe = _apply_rotary_pos_emb_bshd(
                    kv_pe,
                    freqs,
                    mscale=mscale,
                    rotary_interleaved=False,
                    multi_latent_attention=True,
                    mla_output_remove_interleaving=True,
                    high_precision_rope=self.config.high_precision_rope,
                )
                kv_pe = kv_pe.squeeze(2)

                # KV QAT:
                #   kv_nope: bf16 -> fp32 -> fp8e4m3 ->fp32 -> bf16
                if self.use_fp8_qat:
                    kv_nope = fp8_simulate_qat(kv_nope, 64)
                kv = paddle.concat([kv_nope, kv_pe], axis=-1)
        else:
            query = q

        if self.config.swa_high_precision_norm:
            kv = kv.astype(hidden_states.dtype)

        # Single head: key = value = [b, sq, 1, v_head_dim]
        key = kv.unsqueeze(2)
        value = key

        return query, key, value, q_compressed, hidden_states
