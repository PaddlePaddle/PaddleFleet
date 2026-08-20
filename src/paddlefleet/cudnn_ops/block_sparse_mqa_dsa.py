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

"""DSA (FlashMLA sparse fwd + cuDNN DSA bwd) backend for the HySparse
block-sparse MQA gather branch.

DeepSeek-v4's CSA sparse attention (FlashMLA sparse forward + cuDNN DSA
backward) natively handles the absorbed-MQA D=576 query / D_v=512 value
single-shared-head layout. This module bridges the HySparse *block* selection
onto that *token*-level DSA path; the kernel plumbing itself lives in the shared
``paddlefleet.fusions.mqa_sparse_attn.mqa_sparse_attn`` entry (the same FlashMLA
sparse forward + cuDNN DSA backward pair as
``csa_sparse_attn(backend="cudnn")``), which owns the ``h_q == 64`` head padding,
the sinkless ``-1e30`` sink, the asymmetric ``d_v`` and the finite-sink backward
correction. What is specific to HySparse and stays here:

1. **Block -> token index expansion.** Each selected document-relative block
   ``j`` (spanning key cols ``[bos + j*block_B, bos + (j+1)*block_B)``) is
   expanded into its ``block_B`` absolute token columns. ``block_B == 64`` equals
   the SM100 TopK alignment, so one block == one DSA tile chunk (no padding).
2. **Causal/doc masking folded into the index list.** Any expanded column with
   ``col >= eos`` (or belonging to a ``-1`` padding block) is set to ``-1``,
   which DSA treats as invalid -- reproducing the ``valid_range [bos, eos)``
   semantics without a kernel argument.
3. **K==V-unified latent.** DSA takes one ``kv_full`` tensor whose value is its
   leading ``kv_lora_rank`` slice. ``shared_key_sq [B, S, 576]`` already has
   value == leading 512, so it is passed directly and its 576-wide gradient is
   the (combined) gradient w.r.t. the shared latent. ``kv_lora_rank < 512`` is
   re-laid out in ``block_sparse_mqa_attention_dsa``.
"""

from functools import lru_cache

import paddle


@lru_cache(maxsize=1)
def is_dsa_available() -> bool:
    """Whether the FlashMLA sparse fwd + cuDNN DSA bwd path can run here.

    The DSA fwd/bwd kernels are only implemented for SM100+ (Blackwell); there
    is no eager fallback below it. Probe the actual PaddleFleet ops dependencies
    once per process. Avoid importing the standalone ``cudnn`` package here:
    under Paddle's torch proxy its module discovery can recursively enter
    ``find_spec`` and hang the first attention forward.
    """
    try:
        import paddlefleet_ops

        from paddlefleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn

        if paddle.device.cuda.get_device_capability()[0] < 10:
            return False
        if (
            not paddlefleet_ops.is_flash_mla_available()
            or csa_sparse_attn_fwd_cudnn._flash_mla_sparse_fwd is None
        ):
            return False
        if not paddlefleet_ops.is_cudnn_frontend_available():
            return False
    except (ImportError, RuntimeError, AttributeError):
        return False
    return True


def _expand_blocks_to_token_indices(indices, valid_range, block_B):
    """Expand doc-relative block ids to per-token key-column indices.

    Args:
        indices:     [B, S, topk] int, document-relative block ids (-1 padding).
        valid_range: [B, S, 2] int, per-query ``[bos, eos)`` valid key columns.
        block_B:     block size in tokens.

    Returns:
        [B, S, topk * block_B] int32 absolute key columns; entries whose column
        is ``>= eos`` or that belong to a ``-1`` padding block are set to -1.

    The whole computation is a pure integer index construction and MUST NOT
    carry an autograd graph. Under full-layer recompute, ``indices`` /
    ``valid_range`` are recomputed with grad tracking enabled, so the trailing
    ``paddle.where(...).astype("int32")`` would otherwise build a stray
    Where/Cast grad chain. Passing that grad-tracked integer tensor into the
    sparse-attention PyLayer registers a backward edge to an orphan
    ``CastGradNode`` which the engine then schedules with an empty grad holder
    (ref_cnt 0) -> ``cast()`` on an undefined tensor -> segfault. Building the
    indices under ``no_grad`` (and returning a detached tensor) removes the
    stray grad nodes entirely.
    """
    with paddle.no_grad():
        b, s, topk = indices.shape
        bos = valid_range[..., 0:1].astype("int64")  # [B, S, 1]
        eos = valid_range[..., 1:2].astype("int64")  # [B, S, 1]

        blk = indices.astype("int64")  # [B, S, topk]
        start = bos + blk * block_B  # [B, S, topk] absolute col of block start
        offs = paddle.arange(block_B, dtype="int64").reshape([1, 1, 1, block_B])
        cols = start.unsqueeze(-1) + offs  # [B, S, topk, block_B]
        cols = cols.reshape([b, s, topk * block_B])

        blk_invalid = (
            (blk < 0).unsqueeze(-1).expand([b, s, topk, block_B])
        ).reshape([b, s, topk * block_B])
        # cols >= bos always holds (blk >= 0 => start >= bos, offs >= 0); only
        # the tail past eos needs masking, plus columns from -1 padding blocks.
        col_invalid = cols >= eos  # eos broadcasts over the last dim
        invalid = paddle.logical_or(blk_invalid, col_invalid)
        neg = paddle.full_like(cols, -1)
        token_indices = paddle.where(invalid, neg, cols).astype("int32")
    token_indices.stop_gradient = True
    return token_indices


def block_sparse_mqa_attention_dsa(
    query,
    shared_key_sq,
    shared_block_indices,
    valid_range,
    sm_scale=None,
    block_B=64,
    kv_lora_rank=512,
    attn_sink=None,
):
    """HySparse block-sparse gather attention over the absorbed MQA latent.

    Args:
        query:                [B, S, H, Dk] (Dk = kv_lora_rank + rope, e.g. 576).
        shared_key_sq:        [B, S, Dk] shared K/V latent; value = leading
                              ``kv_lora_rank`` slice.
        shared_block_indices: [B, S, topk] int, document-relative selected block
                              ids (-1 padding), from ``select_topk_blocks``.
        valid_range:          [B, S, 2] int, per-query ``[bos, eos)``.
        sm_scale:             softmax scale (defaults to ``Dk ** -0.5``).
        block_B:              block size in tokens (must equal the DSA TopK
                              alignment, 64 on SM100).
        kv_lora_rank:         value dim ``Dv`` (leading slice of the latent).
        attn_sink:            [H] fp32 per-head learnable attention-sink logit,
                              or ``None`` for HySparse's default sinkless softmax
                              (a ``-1e30`` sink whose gradient is discarded).

    Returns:
        ``(out, None)`` where ``out`` is ``[B, S, H * kv_lora_rank]`` and carries
        gradient to ``query`` and ``shared_key_sq`` (and ``attn_sink`` when a
        learnable sink is supplied). The second element keeps the ``(out, lse)``
        tuple shape of the consumer call site; DSA does not surface a
        differentiable lse here.

    ``kv_lora_rank`` < 512 (e.g. ernielite's 448): the FlashMLA sparse kernel
    hard-requires ``d_v == 512`` and ``d_qk in {512, 576}``. We map the smaller
    latent onto that fixed layout by zero-padding the *value* region up to 512:
    the latent ``[value(kv_lora_rank) | rope]`` is re-laid as
    ``[value(kv_lora_rank) | zeros(512 - kv_lora_rank) | rope]`` for both query
    and key. The dot-product score is unchanged (the inserted zeros contribute
    nothing), the kernel value becomes the leading 512 (= real value padded with
    zeros), and the output's leading ``kv_lora_rank`` columns are sliced back out.
    The pad/slice run outside the PyLayer so autograd routes the gradients back
    to the original ``query`` / ``shared_key_sq`` shapes automatically.
    """
    from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

    if sm_scale is None:
        sm_scale = query.shape[-1] ** -0.5
    token_indices = _expand_blocks_to_token_indices(
        shared_block_indices, valid_range, block_B
    )

    b, q_s, num_heads, d_qk = query.shape
    kv_s = shared_key_sq.shape[1]
    pad_v = 512 - kv_lora_rank
    if pad_v > 0:
        # Re-lay [value | rope] -> [value | zeros | rope] so value == leading 512.
        q_val, q_rope = query[..., :kv_lora_rank], query[..., kv_lora_rank:]
        k_val = shared_key_sq[..., :kv_lora_rank]
        k_rope = shared_key_sq[..., kv_lora_rank:]
        zq = paddle.zeros([b, q_s, num_heads, pad_v], dtype=query.dtype)
        zk = paddle.zeros([b, kv_s, pad_v], dtype=shared_key_sq.dtype)
        query_p = paddle.concat([q_val, zq, q_rope], axis=-1)
        key_p = paddle.concat([k_val, zk, k_rope], axis=-1)
        eff_d_v = 512
    elif pad_v < 0:
        raise ValueError(
            f"HySparse DSA supports kv_lora_rank <= 512, got {kv_lora_rank}."
        )
    else:
        query_p, key_p, eff_d_v = query, shared_key_sq, kv_lora_rank

    # ``sink_grad_fusion`` is deliberately left at its default: the fused Triton
    # sink-gradient epilogue (config ``dsa_sink_grad_fusion``) is scoped to
    # MQALatentAttention, so HySparse keeps the eager epilogue even when a
    # learnable ``attn_sink`` is supplied.
    out = mqa_sparse_attn(
        query_p,
        key_p,
        token_indices,
        float(sm_scale),
        int(eff_d_v),
        attn_sink,
    )

    if pad_v > 0:
        # Drop the padded value columns: keep the real leading kv_lora_rank.
        out = out.reshape([b, q_s, num_heads, eff_d_v])[..., :kv_lora_rank]
        out = out.reshape([b, q_s, num_heads * kv_lora_rank])
    return out, None
