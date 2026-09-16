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

"""Fused SwiGLU + fp8-quant activation helpers (training side).

Two things make the expert activation path incomparable across frameworks:

* This side folds the routing weight into the SwiGLU+quant kernel while the
  inference side applies it after the down GEMM. With pow2/ue8m0 block scales the
  two orders do not cancel, so `inspect_tensor_force_unit_probs` forces the
  weights to 1 while probing.
* With fp8 all-to-all the activation reaching the grouped GEMM is fp8 e4m3 plus a
  blockwise 1x128 scale, which cannot be compared with a bf16 dump directly;
  `dequant_dispatched_hidden_bf16` brings it back to bf16.

File layout: `pre_save_func` views first, then the `post_load_func` inverses, then
the entry points the network definition calls. The expert-activation probes are
themselves plain `inspect_tensor(...)` calls whose two hooks come from this
module; the only entry point left here is the one helper that changes what the
model computes.
"""

from __future__ import annotations

import paddle

from paddlefleet.train_infer_consistent_ops.inspect_util import (
    inspect_tag_enabled,
)
from paddlefleet.train_infer_consistent_ops.permute import (
    scatter_canonical_rows,
)

# ---------------------------------------------------------------------------
# pre_save_func views: fp8 + blockwise scale -> comparable bf16
# ---------------------------------------------------------------------------


def dequant_dispatched_hidden_bf16(hs_out, scale):
    """Dequantize a post-dispatch activation to bf16 for the probe.

    Reuses `fused_act_dequant`, the same kernel the real backward path uses.
    Returns None when nothing comparable can be produced, which makes it safe to
    pass straight to `inspect_tensor(..., pre_save_func=...)`.
    """
    if hs_out is None:
        return None
    if scale is None:
        if hs_out.dtype in (paddle.bfloat16, paddle.float32):
            return hs_out
        return None
    return paddle.incubate.nn.functional.fused_act_dequant(hs_out, scale)


# ---------------------------------------------------------------------------
# post_load_func inverses: comparable bf16 -> the live fp8 + blockwise scale
# ---------------------------------------------------------------------------


def _quant_blockwise(x, use_ue8m0):
    """Quantize bf16 -> (fp8 e4m3, 1x128 block scale) with the fwd recipe."""
    out, out_scale = paddle.incubate.nn.functional.fp8_quant_blockwise(
        x,
        output_scale_transpose=False,
        quant_method="1x128",
        input_transpose=False,
        using_pow2_scale=True,
        using_ue8m0_scale=use_ue8m0,
    )
    return out, paddle.transpose(
        paddle.transpose(out_scale, [1, 0]).contiguous(), [1, 0]
    )


def scatter_dispatched_hidden_bf16(hs_out, scale, canon):
    """Fold canonical rows back into a bf16 view of the dispatched activation.

    The bf16 view is rebuilt here instead of being carried over from the matching
    `pre_save_func`: `dequant_dispatched_hidden_bf16` is a pure function of its
    inputs, so the second call is bit-identical, and keeping both hooks stateless
    is what lets the call site stay a single `inspect_tensor(...)` expression.
    The extra dequant only runs when a dump was really loaded, i.e. in
    train_infer_consistent_inspect mode.

    The caller has to drop the fp8 scale alongside this (the result is bf16), which
    is why the probe passes `(hs_out, scale)` in and gets a pair back out.
    """
    return scatter_canonical_rows(
        dequant_dispatched_hidden_bf16(hs_out, scale), canon
    )


def requant_swiglu_output(o2_fp8, o2_scale, canon, use_ue8m0):
    """Fold canonical rows back in and re-quantize with the forward recipe.

    A loaded dump is bf16 while the down GEMM wants (fp8 e4m3, 1x128 block scale),
    so the pair is rebuilt with the same kernel the forward used rather than
    hand-building the ue8m0 scale layout.
    """
    return _quant_blockwise(
        scatter_dispatched_hidden_bf16(o2_fp8, o2_scale, canon), use_ue8m0
    )


# ---------------------------------------------------------------------------
# Probe entry points
# ---------------------------------------------------------------------------


def inspect_tensor_force_unit_probs(probs, tag):
    """All-ones routing weights while `tag` is being probed, else `probs` as is.

    This is the one helper that really changes what the model computes, so it is
    gated on the probe that needs it rather than on the mode as a whole: `tag` has
    to be live under the ABLATION_TAG_WHITELIST / BLACKLIST filters, otherwise a
    run narrowed to unrelated tags would still rewrite the MoE math. Without it
    the expert tail cannot be compared operator by operator (this side folds the
    weight into the SwiGLU+fp8-quant kernel, the inference side multiplies after
    the down GEMM, and under pow2/ue8m0 block scales the two orders do not
    cancel). The logprobs of such a run are diagnostic output anyway, and nothing
    changes while `ABLATION_INSPECT_TENSOR` is unset.

    Args:
        probs: the live routing weights.
        tag: the probe whose comparability this buys -- `moe_act_quant_output`,
            the fused SwiGLU x probs x quant output in `fp8_utils.py`. The down
            GEMM output (`moe_ffn2_output`) inherits the same weighting, so it is
            only comparable in a run where this tag is live too.
    """
    if probs is None or not inspect_tag_enabled(tag):
        return probs
    return paddle.ones_like(probs)
