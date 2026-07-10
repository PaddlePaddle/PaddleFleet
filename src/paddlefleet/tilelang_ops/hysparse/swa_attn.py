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

"""Causal sliding-window attention (SWA), MQA, on top of the tested block-score
TileLang kernels.

A causal sliding window is just a masking pattern: query token ``t`` attends to
keys ``[max(doc_start, t - W + 1), t + 1)``. That half-open range is exactly the
``valid_range [B, S, 2]`` the block-score kernels already consume, and their
``eos - bos`` (document-tight) early-exit naturally bounds the per-token work to
the window ``W`` instead of the full sequence. So SWA needs **no new kernel** --
we feed a windowed ``valid_range`` (see
:func:`..reference.make_sliding_window_valid_range`) to the block-score
forward/backward and drop the (unused) per-block max-logit output.

Why this matters for the HySparse / MLA stack: FlashAttention (FA2/3/4) exposes
``window_size`` and covers SWA directly for ``head_dim <= 256``. But the
**absorbed-MLA MQA** shape has Dk=576 / Dv=512 (> 256), which FA4 does not
support (the model falls back to an eager dense attention). This TileLang SWA
MQA path fills that gap: a fused windowed flash attention at Dk=576/Dv=512 that
is far cheaper than the eager O(S^2) fallback.
"""

import paddle

from .block_score_attn import block_score_mqa_attn_fwd
from .block_score_attn_bwd import block_score_mqa_bwd_interface


class _SlidingWindowMQAAttn(paddle.autograd.PyLayer):
    """Autograd wrapper: causal SWA with a single shared K/V head (MQA)."""

    @staticmethod
    def forward(ctx, q, k, v, valid_range, sm_scale, block_B):
        out, lse, _ = block_score_mqa_attn_fwd(
            q, k, v, valid_range, sm_scale=sm_scale, block_B=block_B
        )
        ctx.save_for_backward(q, k, v, out, lse, valid_range)
        ctx.sm_scale = sm_scale
        ctx.block_B = block_B
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse, valid_range = ctx.saved_tensor()
        dq, dk, dv = block_score_mqa_bwd_interface(
            q,
            k,
            v,
            out,
            dout,
            lse,
            valid_range,
            sm_scale=ctx.sm_scale,
            block_B=ctx.block_B,
        )
        return dq, dk, dv, None


def sliding_window_mqa_attention(
    q, k, v, valid_range, sm_scale=None, block_B=64
):
    """Causal sliding-window attention, MQA (single shared K/V head).

    Args:
        q:           [B, S, H, D] bf16.
        k, v:        [B, S_kv, D] bf16 shared key/value.
        valid_range: [B, S, 2] int32 windowed range (see
                     :func:`..reference.make_sliding_window_valid_range`).
        sm_scale:    softmax scale; defaults to D**-0.5.
        block_B:     key block size.

    Returns:
        out [B, S, H, D_v] bf16, lse [B, S, H] fp32 (non-differentiable).
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return _SlidingWindowMQAAttn.apply(
        q, k, v, valid_range, float(sm_scale), block_B
    )
