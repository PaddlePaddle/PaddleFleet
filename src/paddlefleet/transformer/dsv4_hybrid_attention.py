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

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle import Tensor, nn
from paddle.distributed.fleet.meta_parallel import LayerSpec, build_spec_layer

from paddlefleet.fp8.qat import fp8_simulate_qat
from paddlefleet.models.common.embeddings.rope_utils import (
    _apply_rotary_pos_emb_bshd,
)

try:
    from paddlefleet_ops import deep_gemm

    _DEEP_GEMM_AVAILABLE = hasattr(deep_gemm, "fp8_einsum")
except (ImportError, RuntimeError):
    deep_gemm = None
    _DEEP_GEMM_AVAILABLE = False
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.models.common.embeddings.yarn_rotary_pos_embedding import (
    YarnRotaryEmbedding,
)
from paddlefleet.tensor_parallel import RecomputeWithoutOutput
from paddlefleet.transformer.attention import Attention
from paddlefleet.transformer.csa_attention import (
    CSADocMaskMetadata,
)

if TYPE_CHECKING:
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.enums import AttnMaskType
    from paddlefleet.transformer.transformer_config import TransformerConfig


def _q_rms_norm(q: Tensor, eps: float, high_precision_norm: bool) -> Tensor:
    """RMS normalization for query (no learnable weight)."""
    if high_precision_norm:
        ori_dtype = q.dtype
        q = q.float()
        q = q * paddle.rsqrt(q.square().mean(-1, keepdim=True) + eps)
        return q.astype(ori_dtype)
    else:
        return q * paddle.rsqrt(q.square().mean(-1, keepdim=True) + eps)


class GroupedOutputFP8(paddle.autograd.PyLayer):
    _logged = False
    _logged_bwd = False
    _dump_count = 0

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, num_groups: int, o_lora_rank: int, fp8_wgrad: bool = True, save_original_input: bool = False):
        assert _DEEP_GEMM_AVAILABLE, "GroupedOutputFP8 requires deep_gemm.fp8_einsum"
        b, sq, _, d = x.shape
        assert d % 128 == 0, "FP8 grouped output requires per-group hidden dim to be divisible by 128"
        assert o_lora_rank % 128 == 0, "FP8 grouped output requires o_lora_rank to be divisible by 128"
        if not GroupedOutputFP8._logged:
            print(f"[fp8_grouped_output] forward: x={list(x.shape)} weight={list(weight.shape)} num_groups={num_groups} o_lora_rank={o_lora_rank} d={d} fp8_wgrad={fp8_wgrad} save_original_input={save_original_input}", flush=True)
            GroupedOutputFP8._logged = True
        weight_bf16 = weight.reshape([num_groups, o_lora_rank, d])
        weight_for_gemm = weight_bf16.transpose([0, 2, 1]).contiguous()

        x_2d = x.reshape([-1, d]).contiguous()
        x_fp8, x_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            x_2d,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
            using_pow2_scale=True,
        )[:2]
        x_fp8 = x_fp8.reshape([b * sq, num_groups, d])
        x_scale = x_scale.reshape([b * sq, num_groups, -1])

        weight_fp8, weight_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            weight_for_gemm.reshape([-1, weight_for_gemm.shape[-1]]),
            output_scale_transpose=False,
            quant_method="128x128",
            input_transpose=False,
            using_pow2_scale=True,
        )[:2]
        weight_fp8 = weight_fp8.reshape([num_groups, d, o_lora_rank])
        weight_scale = weight_scale.reshape([num_groups, d // 128, o_lora_rank // 128])

        out = paddle.empty([b * sq, num_groups, o_lora_rank], dtype=paddle.bfloat16)
        deep_gemm.fp8_einsum(
            "bhd,hdr->bhr",
            (x_fp8, x_scale),
            (weight_fp8, weight_scale),
            out,
            recipe=(1, 128, 128),
        )

        # -------- FP8 shadow bf16 diff (debug only) -----------------------
        # Mirrors the use_fp8_grouped_output=False code path in
        # DSv4HybridAttention (see below in this file):
        #     wo_a_weight = weight.reshape([g, r, d])
        #     out_bf16 = paddle.einsum("...gd,grd->...gr", x, wo_a_weight)
        # Enabled via PADDLEFLEET_FP8_SHADOW_DEBUG=1.
        if os.environ.get("PADDLEFLEET_FP8_SHADOW_DEBUG", "0") == "1":
            with paddle.no_grad():
                # Reference 1: paddle.einsum in bf16 (matches use_fp8_grouped_output=False)
                bf16_ref = paddle.einsum(
                    "bsgd,grd->bsgr", x, weight_bf16
                )  # [b, sq, g, r]
                # Reference 2: deep_gemm.einsum in bf16, using the SAME operand
                # layout as the fp8_einsum call above (bhd,hdr->bhr).
                # This isolates FP8 quantization error (fp8_out vs dg_bf16_ref)
                # from any paddle-side einsum layout/precision quirks
                # (paddle_bf16_ref vs dg_bf16_ref should be ~0).
                dg_bf16_ref = paddle.empty(
                    [b * sq, num_groups, o_lora_rank], dtype=paddle.bfloat16
                )
                x_bf16_bhd = x.reshape([b * sq, num_groups, d]).contiguous()
                deep_gemm.einsum(
                    "bhd,hdr->bhr", x_bf16_bhd, weight_for_gemm, dg_bf16_ref
                )

                out_view = out.reshape([b, sq, num_groups, o_lora_rank])
                out_f = out_view.detach().astype("float64").reshape([-1])
                ref_f = bf16_ref.detach().astype("float64").reshape([-1])
                dg_ref_f = dg_bf16_ref.detach().astype("float64").reshape([-1])

                def _calc_diff(a, b_):
                    denom = (a * a + b_ * b_).sum()
                    if float(denom) == 0.0:
                        return 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
                    sim = 2.0 * (a * b_).sum() / denom
                    d_ = float(1.0 - sim)
                    m_ = float((a - b_).abs().max())
                    n_a = float((a * a).sum().sqrt())
                    n_b = float((b_ * b_).sum().sqrt())
                    c_ = float((a * b_).sum()) / (n_a * n_b + 1e-12)
                    # Range-Relative RMSE (TE tests/pytorch/utils.py:238):
                    #   rmse       = sqrt(mean((a-b)^2))
                    #   rmse_range = max(a,b) - min(a,b)  (union span)
                    #   rrmse      = rmse / rmse_range          (range-normalized)
                    # Standard relative RMSE (comparable with FP8 unit roundoff):
                    #   rrmse_std  = rmse / sqrt(mean(b_^2))   (ref-RMS-normalized)
                    rmse_ = float((a - b_).square().mean().sqrt())
                    hi = float(paddle.maximum(a.max(), b_.max()))
                    lo = float(paddle.minimum(a.min(), b_.min()))
                    rng = hi - lo
                    rr_ = rmse_ / rng if rng > 0.0 else 0.0
                    ref_rms = float(b_.square().mean().sqrt())
                    rr_std_ = rmse_ / ref_rms if ref_rms > 0.0 else 0.0
                    return d_, m_, c_, rmse_, rr_, rr_std_

                # fp8 vs paddle-bf16 (original metric)
                diff, max_abs, cos_sim, rmse, rrmse, rrmse_std = _calc_diff(
                    out_f, ref_f
                )
                # fp8 vs deep_gemm-bf16 (pure fp8 quant error)
                diff_dg, max_abs_dg, cos_sim_dg, rmse_dg, rrmse_dg, rrmse_std_dg = (
                    _calc_diff(out_f, dg_ref_f)
                )
                # paddle-bf16 vs deep_gemm-bf16 (should be ~0 if consistent)
                (
                    diff_ref,
                    max_abs_ref,
                    cos_sim_ref,
                    rmse_ref,
                    rrmse_ref,
                    rrmse_std_ref,
                ) = _calc_diff(ref_f, dg_ref_f)
            print(
                f"[FP8_SHADOW][fwd_grouped_output] "
                f"x={list(x.shape)} weight={list(weight_bf16.shape)} "
                f"out=[{b},{sq},{num_groups},{o_lora_rank}] "
                f"fp8_vs_paddle: diff={diff:.4e} max_abs={max_abs:.4e} cos_sim={cos_sim:.6f} rmse={rmse:.4e} rrmse={rrmse:.4e} rrmse_std={rrmse_std:.4e} | "
                f"fp8_vs_dg: diff={diff_dg:.4e} max_abs={max_abs_dg:.4e} cos_sim={cos_sim_dg:.6f} rmse={rmse_dg:.4e} rrmse={rrmse_dg:.4e} rrmse_std={rrmse_std_dg:.4e} | "
                f"paddle_vs_dg: diff={diff_ref:.4e} max_abs={max_abs_ref:.4e} cos_sim={cos_sim_ref:.6f} rmse={rmse_ref:.4e} rrmse={rrmse_ref:.4e} rrmse_std={rrmse_std_ref:.4e}",
                flush=True,
            )

            # -------- Dump raw bf16 inputs when diff exceeds threshold --------
            # Disabled by default. Enable by setting the dump directory:
            #   PADDLEFLEET_FP8_SHADOW_DUMP_DIR       (unset -> feature off)
            #   PADDLEFLEET_FP8_SHADOW_DUMP_THRESHOLD (default 1e-3)
            #   PADDLEFLEET_FP8_SHADOW_DUMP_MAX       (default 8, per rank)
            # Triggers on max(fp8_vs_paddle, fp8_vs_dg) so any fp8 quant issue is captured.
            _dump_dir = os.environ.get("PADDLEFLEET_FP8_SHADOW_DUMP_DIR")
            if _dump_dir:
                _dump_thr = float(
                    os.environ.get("PADDLEFLEET_FP8_SHADOW_DUMP_THRESHOLD", "1e-3")
                )
                _dump_max = int(
                    os.environ.get("PADDLEFLEET_FP8_SHADOW_DUMP_MAX", "8")
                )
                _trigger_diff = max(diff, diff_dg)
                if (
                    _trigger_diff > _dump_thr
                    and GroupedOutputFP8._dump_count < _dump_max
                ):
                    import numpy as np

                    os.makedirs(_dump_dir, exist_ok=True)
                    try:
                        _rank = paddle.distributed.get_rank()
                    except Exception:
                        _rank = 0
                    _idx = GroupedOutputFP8._dump_count
                    GroupedOutputFP8._dump_count += 1
                    _fname = (
                        f"fwd_grouped_output_rank{_rank}_idx{_idx}"
                        f"_b{b}_sq{sq}_g{num_groups}_r{o_lora_rank}_d{d}"
                        f"_diff{_trigger_diff:.3e}.npz"
                    )
                    _fpath = os.path.join(_dump_dir, _fname)
                    with paddle.no_grad():
                        np.savez(
                            _fpath,
                            x=x.detach().astype("float32").numpy(),
                            weight=weight_bf16.detach()
                            .astype("float32")
                            .numpy(),
                            # keep original dtype string for reproduction fidelity
                            x_dtype=str(x.dtype),
                            weight_dtype=str(weight_bf16.dtype),
                            diff_fp8_vs_paddle=np.float64(diff),
                            diff_fp8_vs_dg=np.float64(diff_dg),
                            diff_paddle_vs_dg=np.float64(diff_ref),
                            # Range-Relative RMSE (per TE tests/pytorch/utils.py)
                            rrmse_fp8_vs_paddle=np.float64(rrmse),
                            rrmse_fp8_vs_dg=np.float64(rrmse_dg),
                            rrmse_paddle_vs_dg=np.float64(rrmse_ref),
                            # Standard relative RMSE (rmse / sqrt(mean(ref^2)))
                            rrmse_std_fp8_vs_paddle=np.float64(rrmse_std),
                            rrmse_std_fp8_vs_dg=np.float64(rrmse_std_dg),
                            rrmse_std_paddle_vs_dg=np.float64(rrmse_std_ref),
                            rmse_fp8_vs_paddle=np.float64(rmse),
                            rmse_fp8_vs_dg=np.float64(rmse_dg),
                            rmse_paddle_vs_dg=np.float64(rmse_ref),
                        )
                    print(
                        f"[FP8_SHADOW][fwd_grouped_output][DUMP] rank={_rank} "
                        f"idx={_idx}/{_dump_max} diff={_trigger_diff:.4e} > {_dump_thr:.1e} "
                        f"saved to {_fpath}",
                        flush=True,
                    )

        # FP8 opt-in policy: when ``save_original_input=False`` we skip
        # stashing the bf16 activation on ctx to save memory. Since the wgrad
        # path re-quantizes x with quant_method="1x128" anyway, we run the same
        # quantization eagerly here and stash the fp8 result on ctx. This is
        # numerically identical to re-quantizing in backward (deterministic).
        # For ``fp8_wgrad=False`` the backward bf16 wgrad path dequantizes
        # ``x_fp8_w_pre`` back to bf16 on the fly.
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
            x_wgrad_pre = (
                x.reshape([b * sq, num_groups, d])
                .transpose([1, 2, 0])
                .contiguous()
            )  # [h, d, b*sq]
            x_w_2d_pre = x_wgrad_pre.reshape([-1, b * sq]).contiguous()
            x_fp8_w_pre, x_scale_w_pre = (
                paddle.incubate.nn.functional.fp8_quant_blockwise(
                    x_w_2d_pre,
                    output_scale_transpose=False,
                    quant_method="1x128",
                    input_transpose=False,
                    using_pow2_scale=True,
                )[:2]
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
        print(f"[fp8_grouped_output] backward: grad_output={list(grad_output.shape)} x_saved={ctx.bf16_x_saved} fp8_wgrad={fp8_wgrad}", flush=True) if not GroupedOutputFP8._logged_bwd else None
        GroupedOutputFP8._logged_bwd = True

        # dgrad: always FP8
        # grad_x = grad_output @ weight, "bhr,hdr->bhd"
        # weight is [g, r, d], need [g, d, r] for "hdr" layout
        grad_out_2d = grad_output.reshape([-1, o_lora_rank]).contiguous()
        go_fp8, go_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            grad_out_2d,
            output_scale_transpose=False,
            quant_method="1x128",
            input_transpose=False,
            using_pow2_scale=True,
        )[:2]
        go_fp8 = go_fp8.reshape([b * sq, num_groups, o_lora_rank])
        go_scale = go_scale.reshape([b * sq, num_groups, -1])

        w_for_dgrad = weight.transpose([0, 2, 1]).contiguous()  # [g, r, d] -> [g, d, r]
        w_2d = w_for_dgrad.reshape([-1, o_lora_rank]).contiguous()  # [g*d, r]
        w_fp8, w_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
            w_2d,
            output_scale_transpose=False,
            quant_method="128x128",
            input_transpose=False,
            using_pow2_scale=True,
        )[:2]
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

        # -------- FP8 shadow bf16 diff for dgrad (debug only) -------------
        # Mirrors the bf16-equivalent of the fp8 dgrad:
        #     grad_x[b,s,g,d] = sum_r grad_output[b,s,g,r] * weight[g,r,d]
        if os.environ.get("PADDLEFLEET_FP8_SHADOW_DEBUG", "0") == "1":
            with paddle.no_grad():
                bf16_ref = paddle.einsum(
                    "bsgr,grd->bsgd", grad_output, weight
                )
                out_f = grad_x.detach().astype("float64").reshape([-1])
                ref_f = bf16_ref.detach().astype("float64").reshape([-1])
                denom = (out_f * out_f + ref_f * ref_f).sum()
                if float(denom) == 0.0:
                    diff = 0.0
                else:
                    sim = 2.0 * (out_f * ref_f).sum() / denom
                    diff = float(1.0 - sim)
                max_abs = float((out_f - ref_f).abs().max())
                norm_out = float((out_f * out_f).sum().sqrt())
                norm_ref = float((ref_f * ref_f).sum().sqrt())
                dot = float((out_f * ref_f).sum())
                cos_sim = dot / (norm_out * norm_ref + 1e-12)
                # Range-Relative RMSE (TE tests/pytorch/utils.py:238)
                rmse = float((out_f - ref_f).square().mean().sqrt())
                _hi = float(paddle.maximum(out_f.max(), ref_f.max()))
                _lo = float(paddle.minimum(out_f.min(), ref_f.min()))
                _rng = _hi - _lo
                rrmse = rmse / _rng if _rng > 0.0 else 0.0
                # Standard relative RMSE: rmse / sqrt(mean(ref^2))
                _ref_rms = float(ref_f.square().mean().sqrt())
                rrmse_std = rmse / _ref_rms if _ref_rms > 0.0 else 0.0
            print(
                f"[FP8_SHADOW][bwd_grouped_output_dgrad] "
                f"grad_output={list(grad_output.shape)} weight={list(weight.shape)} "
                f"grad_x={list(grad_x.shape)} "
                f"diff={diff:.4e} max_abs={max_abs:.4e} cos_sim={cos_sim:.6f} "
                f"rmse={rmse:.4e} rrmse={rrmse:.4e} rrmse_std={rrmse_std:.4e}",
                flush=True,
            )

        # wgrad
        if fp8_wgrad:
            # grad_weight = x^T @ grad_output, "bhd,bhr->hdr"
            # DeepGEMM internally permutes A,B by {1,2,0} -> [h,d,b] @ [h,r,b]^T
            # fp8_bmm with recipe=(1,1,128) expects A_scale: [h, d, ceil_div(b,128)]
            # So we need to quant along the b dimension (last dim after permute)
            # Approach: permute to [h,d,b], quant 2D [h*d, b] with 1x128, get scale [h*d, b//128]
            assert (b * sq) % 128 == 0, (
                f"FP8 grouped output wgrad requires (batch*seqlen) to be divisible by 128, "
                f"got b={b}, sq={sq}, b*sq={b * sq}"
            )
            if ctx.bf16_x_saved:
                x_wgrad = x.reshape([b * sq, num_groups, d]).transpose([1, 2, 0]).contiguous()  # [h, d, b*sq]
                x_w_2d = x_wgrad.reshape([-1, b * sq]).contiguous()  # [h*d, b*sq]
                x_fp8_w, x_scale_w = paddle.incubate.nn.functional.fp8_quant_blockwise(
                    x_w_2d,
                    output_scale_transpose=False,
                    quant_method="1x128",
                    input_transpose=False,
                    using_pow2_scale=True,
                )[:2]
                # scale: [h*d, ceil(b*sq/128)] -> reshape to [h, d, ceil(b*sq/128)]
                x_fp8_w = x_fp8_w.reshape([num_groups, d, b * sq])
                x_scale_w = x_scale_w.reshape([num_groups, d, -1])
            else:
                # Reuse fp8 activation pre-computed in forward (identical
                # quantization to the bf16-saved path, so precision matches).
                x_fp8_w = ctx.x_fp8_w_pre
                x_scale_w = ctx.x_scale_w_pre

            go_wgrad = grad_output.reshape([b * sq, num_groups, o_lora_rank]).transpose([1, 2, 0]).contiguous()  # [h, r, b*sq]
            go_w_2d = go_wgrad.reshape([-1, b * sq]).contiguous()  # [h*r, b*sq]
            go_fp8_w, go_scale_w = paddle.incubate.nn.functional.fp8_quant_blockwise(
                go_w_2d,
                output_scale_transpose=False,
                quant_method="1x128",
                input_transpose=False,
                using_pow2_scale=True,
            )[:2]
            go_fp8_w = go_fp8_w.reshape([num_groups, o_lora_rank, b * sq])
            go_scale_w = go_scale_w.reshape([num_groups, o_lora_rank, -1])

            # Now pass pre-permuted tensors. But fp8_einsum will permute {1,2,0} again!
            # We need to pass in the ORIGINAL [b,h,d] layout so kernel can permute internally.
            # Re-permute back: [h,d,b] -> [b,h,d]
            x_fp8_w = x_fp8_w.transpose([2, 0, 1]).contiguous()  # [b*sq, h, d]
            x_scale_w = x_scale_w.transpose([2, 0, 1]).contiguous()  # [ceil(b/128), h, d]
            go_fp8_w = go_fp8_w.transpose([2, 0, 1]).contiguous()  # [b*sq, h, r]
            go_scale_w = go_scale_w.transpose([2, 0, 1]).contiguous()  # [ceil(b/128), h, r]

            grad_weight = paddle.empty([num_groups, d, o_lora_rank], dtype=paddle.bfloat16)
            deep_gemm.fp8_einsum(
                "bhd,bhr->hdr",
                (x_fp8_w, x_scale_w),
                (go_fp8_w, go_scale_w),
                grad_weight,
                recipe=(1, 1, 128),
            )
            grad_weight = grad_weight.transpose([0, 2, 1]).contiguous()  # [g, d, r] -> [g, r, d]

            # -------- FP8 shadow bf16 diff for wgrad (debug only) ---------
            # Mirrors the fp8_wgrad=False branch below:
            #     grad_weight = paddle.einsum("bsgd,bsgr->grd", x, grad_output)
            if os.environ.get("PADDLEFLEET_FP8_SHADOW_DEBUG", "0") == "1" and x is not None:
                with paddle.no_grad():
                    bf16_ref = paddle.einsum(
                        "bsgd,bsgr->grd", x, grad_output
                    )
                    out_f = grad_weight.detach().astype("float64").reshape([-1])
                    ref_f = bf16_ref.detach().astype("float64").reshape([-1])
                    denom = (out_f * out_f + ref_f * ref_f).sum()
                    if float(denom) == 0.0:
                        diff = 0.0
                    else:
                        sim = 2.0 * (out_f * ref_f).sum() / denom
                        diff = float(1.0 - sim)
                    max_abs = float((out_f - ref_f).abs().max())
                    norm_out = float((out_f * out_f).sum().sqrt())
                    norm_ref = float((ref_f * ref_f).sum().sqrt())
                    dot = float((out_f * ref_f).sum())
                    cos_sim = dot / (norm_out * norm_ref + 1e-12)
                    # Range-Relative RMSE (TE tests/pytorch/utils.py:238)
                    rmse = float((out_f - ref_f).square().mean().sqrt())
                    _hi = float(paddle.maximum(out_f.max(), ref_f.max()))
                    _lo = float(paddle.minimum(out_f.min(), ref_f.min()))
                    _rng = _hi - _lo
                    rrmse = rmse / _rng if _rng > 0.0 else 0.0
                    # Standard relative RMSE: rmse / sqrt(mean(ref^2))
                    _ref_rms = float(ref_f.square().mean().sqrt())
                    rrmse_std = rmse / _ref_rms if _ref_rms > 0.0 else 0.0
                print(
                    f"[FP8_SHADOW][bwd_grouped_output_wgrad] "
                    f"x={list(x.shape)} grad_output={list(grad_output.shape)} "
                    f"grad_weight={list(grad_weight.shape)} "
                    f"diff={diff:.4e} max_abs={max_abs:.4e} cos_sim={cos_sim:.6f} "
                    f"rmse={rmse:.4e} rrmse={rrmse:.4e} rrmse_std={rrmse_std:.4e}",
                    flush=True,
                )
        else:
            # bf16 path
            if x is None:
                # save_original_input=False: dequant the pre-quantized fp8
                # activation back to bf16. ``x_fp8_w_pre`` is
                # ``[g, d, b*sq]``; reshape to 2D ``[g*d, b*sq]`` for
                # fused_act_dequant, then unpack back to ``[b, sq, g, d]``.
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

        return grad_x, grad_weight.reshape([num_groups * o_lora_rank, d])


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
        # Per-layer RoPE (potentially different base for compressed layers)
        rope_base = getattr(config, "rotary_base", 10000)
        if compress_ratio > 1:
            rope_base = config.csa_compress_rotary_base

        use_compressed_yarn = compress_ratio > 1
        if not use_compressed_yarn:
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=getattr(config, "rotary_percent", 1.0),
                rotary_base=rope_base,
            )
        else:
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
        # FP8 comes from ``config`` now. ``save_original_input`` is a settable
        # attribute (default ``not self.fp8``) so callers can override it
        # post-construction if the wgrad path needs the bf16 activation kept.
        # Assert TP=1 when fp8 is enabled — DSv4 FP8 is currently only wired up
        # for the non-parallel path.
        self.use_fp8_qat = getattr(config, "use_fp8_qat", False)
        self.fp8 = bool(getattr(config, "fp8", False))
        self.fp8_wgrad = bool(getattr(config, "fp8_wgrad", False)) and self.fp8
        self.save_original_input = not self.fp8
        self.use_fp8_grouped_output = self.fp8 and _DEEP_GEMM_AVAILABLE
        if self.fp8:
            tp_nranks = getattr(self.pg_collection.tp, "nranks", 1)
            assert tp_nranks == 1, (
                "DSv4HybridAttention FP8 currently requires TP=1, "
                f"got tp.nranks={tp_nranks}"
            )
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
            and config.recompute_modules is not None
            and "gated_attn" in config.recompute_modules
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
        b, sq, _ = hidden_states.shape
        position_offset = cp_rank * sq if cp_size > 1 else 0

        docmask_meta = None
        ratio = int(getattr(self.core_attention, "compress_ratio", 0))
        if startend_row_indices is not None:
            docmask_seqlen = sq * cp_size if cp_size > 1 else sq
            docmask_meta = CSADocMaskMetadata.build(
                max(1, ratio),
                b,
                docmask_seqlen,
                startend_row_indices,
                dense_mode=self.config.csa_dense_mode,
            )

        query, key, value, q_compressed, kv_compressed = (
            self.get_query_key_value_tensors(
                hidden_states=hidden_states,
                position_offset=position_offset,
                docmask_meta=docmask_meta,
            )
        )

        # Core attention (CompressedSparseAttention)
        input_ids = kwargs.get("input_ids", None)
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            x=hidden_states,
            qr=q_compressed,
            input_ids=input_ids,
            docmask_meta=docmask_meta,
        )
        # core_attn_out: [b, sq, np * v_head_dim]

        # Inverse RoPE on last qk_pos_emb_head_dim of each head
        b, sq, _ = core_attn_out.shape
        pos_dim = self.qk_pos_emb_head_dim
        nope_dim = self.v_head_dim - pos_dim

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

            if (
                self.config.apply_rope_fusion
                and not self.config.high_precision_rope
            ):
                from paddlefleet.triton_ops import fused_apply_mla_rope_inplace

                # The clone is necessary because sparse attention depends on core_attn_out
                # for backward. However, it is still 10x faster than the unfused path.
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

        # Grouped output projection
        core_attn_out = core_attn_out.reshape([b, sq, self.o_local_groups, -1])
        if self.use_fp8_grouped_output:
            core_attn_out = GroupedOutputFP8.apply(
                core_attn_out,
                self.linear_o_group_proj,
                self.o_local_groups,
                self.config.o_lora_rank,
                self.fp8_wgrad,
                self.save_original_input,
            )
        else:
            wo_a_weight = self.linear_o_group_proj.reshape(
                [self.o_local_groups, self.config.o_lora_rank, -1]
            )
            core_attn_out = paddle.einsum(
                "...gd,grd->...gr", core_attn_out, wo_a_weight
            )
            core_attn_out = core_attn_out.reshape([b, sq, -1])

        # Apply gated attention
        if self.gated_attention:
            # Gate input source: q_compressed (post q_layernorm, dim=q_lora_rank)
            # when gated_attn_use_q_lora is set, otherwise hidden_states.
            gate_source = (
                q_compressed if self.gated_attn_use_q_lora else hidden_states
            )
            if self.recompute_gated_attn:
                gate_recompute = RecomputeWithoutOutput()
                core_attn_out = gate_recompute.recompute(
                    self._gate,
                    gate_source,
                    core_attn_out,
                    preserve_rng_state=False,
                    share_grad_holder=True,
                )
            else:
                core_attn_out = self._gate(gate_source, core_attn_out)

        # Output projection
        output, bias = self.o_proj(core_attn_out)

        if self.gated_attention and self.recompute_gated_attn:
            gate_recompute.discard_output_and_register_recompute(output)

        return output, bias

    def _gate(self, gate_source: Tensor, core_attn_out: Tensor) -> Tensor:
        gate, _ = self.gate_proj(gate_source)
        if getattr(self.config, "sigmoid_gate_fusion", False):
            from paddlefleet.triton_ops import SigmoidGateFusionTriton

            core_attn_out = SigmoidGateFusionTriton.apply(core_attn_out, gate)
        else:
            core_attn_out = core_attn_out * paddle.nn.functional.sigmoid(gate)
        return core_attn_out

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
            config.q_lora_rank,
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
            hidden_size=config.q_lora_rank,
            eps=getattr(config, "rms_norm_eps", 1e-5),
        )

        # Q up projection: q_lora_rank -> num_heads * q_head_dim (column parallel)
        self.linear_q_up_proj = build_spec_layer(
            sublayers_spec.linear_q_up_proj,
            config.q_lora_rank,
            self.num_attention_heads * q_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        # KV projection: hidden_size -> v_head_dim (single head)
        self.linear_kv_proj = build_spec_layer(
            sublayers_spec.linear_kv_proj,
            config.hidden_size,
            config.v_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.pg_collection.tp,
        )

        # KV layernorm
        self.kv_layernorm = build_spec_layer(
            sublayers_spec.kv_layernorm,
            config=config,
            hidden_size=config.v_head_dim,
            eps=getattr(config, "rms_norm_eps", 1e-5),
        )

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
        q_compressed, _ = self.linear_q_down_proj(
            hidden_states
        )  # [b, sq, q_lora_rank]
        q_compressed = self.q_layernorm(q_compressed)

        q, _ = self.linear_q_up_proj(q_compressed)  # [b, sq, n * v_head_dim]
        q = q.reshape([b, sq, self.num_attention_heads, self.v_head_dim])
        q = _q_rms_norm(
            q,
            getattr(self.config, "rms_norm_eps", 1e-5),
            high_precision_norm=self.config.swa_high_precision_norm,
        )

        # KV path
        kv, _ = self.linear_kv_proj(hidden_states)  # [b, sq, v_head_dim]

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
