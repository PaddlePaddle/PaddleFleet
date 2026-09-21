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
Triton kernels for efficient QK attention statistics computation.
Includes: Max Logits, Mean Logits, Attention Entropy, and Attention Sink Weights.
"""

import logging

import torch

from ...core.triton_qk_kernel import qk_stats_kernel, qk_stats_packed_kernel

logger = logging.getLogger(__name__)


def compute_qk_stats_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = True,
    attn_sink: torch.Tensor | None = None,
) -> dict:
    """
    Compute QK statistics using optimized Triton kernel.
    Input: [B, H, S, D] (already permuted by compute_qk_stats).
    Returns: Max Logits, Mean Logits, Entropy, Sink Weights.

    ``attn_sink``: optional per-query-head sink logit ``[H]`` folded into the
    softmax denominator (``None`` = vanilla, real-key-only).
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    # Output tensors [batch, num_heads]
    max_logits = torch.empty(
        (batch, num_heads), device=q.device, dtype=torch.float32
    )
    mean_logits = torch.empty(
        (batch, num_heads), device=q.device, dtype=torch.float32
    )
    entropy = torch.empty(
        (batch, num_heads), device=q.device, dtype=torch.float32
    )
    sink = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)
    count = torch.empty(
        (batch, num_heads), device=q.device, dtype=torch.float32
    )

    grid = (batch * num_heads,)

    # Tuning block sizes for performance
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    if head_dim > 64:
        BLOCK_K = 128

    # None is not a valid kernel pointer: pass a dummy zero buffer, gate on HAS_SINK.
    sink_present = attn_sink is not None
    sink_buf = (
        attn_sink
        if sink_present
        else torch.zeros(num_heads, device=q.device, dtype=torch.float32)
    )

    qk_stats_kernel[grid](
        q,
        k,
        sink_buf,
        max_logits,
        mean_logits,
        entropy,
        sink,
        count,
        batch,
        num_heads,
        seq_len,  # seq_len_q
        seq_len,  # seq_len_k: symmetric (no CP sharding in this monitor path)
        head_dim,
        1,  # heads_per_group: k is already repeat_interleave-expanded above
        0,  # q_row_offset: 0 without CP (Q is the full local sequence)
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        max_logits.stride(0),
        max_logits.stride(1),
        scale=scale,
        apply_causal_mask=causal,
        HAS_SINK=sink_present,
        ROW_STRIDE=1,  # full-sequence (exact) behavior, unchanged
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return {
        "max_per_head": max_logits,
        "mean_per_head": mean_logits,
        "entropy_per_head": entropy,
        "sink_per_head": sink,
        "max_global": max_logits.max(),
        "mean_global": mean_logits.mean(),
        "entropy_global": entropy.mean(),
        "sink_global": sink.mean(),
    }


def compute_qk_stats_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = True,
    attn_sink: torch.Tensor | None = None,
) -> dict:
    """
    Reference PyTorch implementation including Entropy and Sink.
    Input: [B, H, S, D] (already permuted by compute_qk_stats).

    ``attn_sink``: optional per-query-head sink logit ``[H]`` folded in as one
    extra key-less softmax column (excluded from max/mean and the sink numerator).
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    # 1. Logits
    # [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        logits.masked_fill_(mask, float("-inf"))

    # 2. Stats: Max & Mean (Raw Logits)
    # Filter out -inf for mean calculation
    valid_mask = logits > -1e9
    max_per_head = logits.max(dim=-1)[0].max(dim=-1)[0]

    logits_zeroed = torch.where(
        valid_mask, logits, torch.tensor(0.0, device=q.device)
    )
    sum_logits = logits_zeroed.sum(dim=(-2, -1))
    count = valid_mask.sum(dim=(-2, -1))
    mean_per_head = sum_logits / count.clamp(min=1)

    # 3. Softmax & Entropy & Sink
    # Fold the sink logit in as one extra key-less column: it enters the softmax
    # denominator and entropy (it's part of the model's distribution), but not
    # max/mean or the token-0 sink numerator (those describe real keys).
    if attn_sink is not None:
        sink_col = (
            attn_sink.reshape(1, num_heads, 1, 1)
            .expand(batch, num_heads, seq_len, 1)
            .to(logits.dtype)
        )
        ext_logits = torch.cat([logits, sink_col], dim=-1)  # [B, H, S, S+1]
        ext_valid = torch.cat(
            [valid_mask, torch.ones_like(sink_col, dtype=torch.bool)], dim=-1
        )
    else:
        ext_logits = logits
        ext_valid = valid_mask

    probs = torch.softmax(ext_logits, dim=-1)  # [B, H, S, S(+1)]
    log_probs = torch.log_softmax(ext_logits, dim=-1)
    entropy_map = -(probs * log_probs)
    entropy_map = torch.where(
        ext_valid, entropy_map, torch.tensor(0.0, device=q.device)
    )
    row_entropy = entropy_map.sum(dim=-1)  # [B, H, S]
    avg_entropy = row_entropy.mean(dim=-1)  # [B, H]

    # Sink: probability of real token 0 (never the appended sink column)
    sink_probs = probs[..., 0]  # [B, H, S]
    avg_sink = sink_probs.mean(dim=-1)  # [B, H]

    return {
        "max_per_head": max_per_head,
        "mean_per_head": mean_per_head,
        "entropy_per_head": avg_entropy,
        "sink_per_head": avg_sink,
        "max_global": max_per_head.max(),
        "mean_global": mean_per_head.mean(),
        "entropy_global": avg_entropy.mean(),
        "sink_global": avg_sink.mean(),
    }


def compute_qk_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool = True,
    use_triton: bool = True,
    attn_sink: torch.Tensor | None = None,
) -> dict:
    """
    Unified entry point. Input layout: [S, B, H, D] (Megatron core_attention convention).
    Permutes to [B, H, S, D] before dispatching to backend.

    ``attn_sink``: optional per-query-head sink logit ``[H]`` (softmax_offset).
    """
    seq_len, batch, num_q_heads, head_dim = q.shape
    _, _, num_k_heads, _ = k.shape

    if num_q_heads != num_k_heads:
        heads_per_group = num_q_heads // num_k_heads
        k = k.repeat_interleave(heads_per_group, dim=2)

    # Permute [S, B, H, D] -> [B, H, S, D] for both backends
    q = q.permute(1, 2, 0, 3).contiguous()
    k = k.permute(1, 2, 0, 3).contiguous()

    if use_triton:
        return compute_qk_stats_triton(q, k, causal, attn_sink=attn_sink)

    return compute_qk_stats_pytorch(q, k, causal, attn_sink=attn_sink)


def _cu_seqlens_to_token_arrays(
    cu_seqlens: torch.Tensor, total_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map ``cu_seqlens`` to per-token sequence boundaries, entirely on-device.

    Given ``cu_seqlens = [0, l0, l0+l1, ...]`` (the packed cumulative sequence
    lengths), returns two int32 tensors of shape ``[total_tokens]``:
      * ``seq_start[t]`` = start token index of the sequence containing token t
      * ``seq_end[t]``   = end (exclusive) token index of that sequence

    Padding tokens (``t >= cu_seqlens[-1]``) get an empty ``[0, 0)`` range so
    the kernel's ``same_seq`` mask discards them.

    Sync-free: built from ``torch.searchsorted`` + ``torch.where`` with NO
    ``.item()`` / ``.cpu()`` / ``.tolist()``, so it is safe to call from a
    forward hook without breaking compute/comm overlap.
    """
    device = cu_seqlens.device
    cu = cu_seqlens.to(torch.int64)
    num_seqs = cu.numel() - 1
    token_ids = torch.arange(total_tokens, device=device, dtype=torch.int64)

    # seq_idx: i such that cu[i] <= t < cu[i+1]
    seq_idx = torch.searchsorted(cu, token_ids, right=True) - 1
    seq_idx = seq_idx.clamp(0, max(num_seqs - 1, 0))

    seq_start = cu[seq_idx]
    seq_end = cu[seq_idx + 1]

    # Padding tokens beyond the last real token -> empty range.
    last = cu[-1]
    is_pad = token_ids >= last
    zero = torch.zeros((), device=device, dtype=torch.int64)
    seq_start = torch.where(is_pad, zero, seq_start)
    seq_end = torch.where(is_pad, zero, seq_end)

    return seq_start.to(torch.int32).contiguous(), seq_end.to(
        torch.int32
    ).contiguous()


def compute_qk_stats_triton_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    causal: bool = True,
    attn_sink: torch.Tensor | None = None,
) -> dict:
    """Packed (THD) QK statistics via the split-M Triton kernel.

    Input: [H, T, D] (GQA already repeat_interleave-expanded, batch flattened).
    Reduces the per-M-block partial buffers into per-head stats using row-first
    mean semantics (mean over valid rows of each row's mean logit), matching
    the dense ``qk_stats_partial_kernel`` reduction.

    ``attn_sink``: optional per-query-head sink logit ``[H]`` folded into the
    softmax denominator on top of the per-sequence sink column.
    """
    num_heads, total_tokens, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    seq_start, seq_end = _cu_seqlens_to_token_arrays(cu_seqlens, total_tokens)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    if head_dim > 64:
        BLOCK_K = 128

    num_m_blocks = (total_tokens + BLOCK_M - 1) // BLOCK_M

    def _buf():
        return torch.empty(
            (num_heads, num_m_blocks), device=q.device, dtype=torch.float32
        )

    p_max = _buf()
    p_sum_logit = _buf()
    p_sum_row_mean = _buf()
    p_count = _buf()
    p_sum_entropy = _buf()
    p_sum_sink = _buf()
    p_valid_rows = _buf()

    grid = (num_heads, num_m_blocks)

    # None is not a valid kernel pointer: pass a dummy zero buffer, gate on HAS_SINK.
    sink_present = attn_sink is not None
    sink_buf = (
        attn_sink
        if sink_present
        else torch.zeros(num_heads, device=q.device, dtype=torch.float32)
    )

    qk_stats_packed_kernel[grid](
        q,
        k,
        sink_buf,
        seq_start,
        seq_end,
        p_max,
        p_sum_logit,
        p_sum_row_mean,
        p_count,
        p_sum_entropy,
        p_sum_sink,
        p_valid_rows,
        num_heads,
        total_tokens,
        head_dim,
        num_m_blocks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        p_max.stride(0),
        p_max.stride(1),
        scale=scale,
        apply_causal_mask=causal,
        HAS_SINK=sink_present,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Reduce partials over the M-block dim -> per-head [H].
    valid_rows = p_valid_rows.sum(dim=1)
    safe_rows = valid_rows.clamp(min=1)

    max_per_head = p_max.max(dim=1).values
    mean_per_head = p_sum_row_mean.sum(dim=1) / safe_rows
    entropy_per_head = p_sum_entropy.sum(dim=1) / safe_rows
    sink_per_head = p_sum_sink.sum(dim=1) / safe_rows

    # Shape [1, H] to match the dense [B, H] convention consumed by the hook.
    max_per_head = max_per_head.unsqueeze(0)
    mean_per_head = mean_per_head.unsqueeze(0)
    entropy_per_head = entropy_per_head.unsqueeze(0)
    sink_per_head = sink_per_head.unsqueeze(0)

    return {
        "max_per_head": max_per_head,
        "mean_per_head": mean_per_head,
        "entropy_per_head": entropy_per_head,
        "sink_per_head": sink_per_head,
        "max_global": max_per_head.max(),
        "mean_global": mean_per_head.mean(),
        "entropy_global": entropy_per_head.mean(),
        "sink_global": sink_per_head.mean(),
    }


def compute_qk_stats_pytorch_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    causal: bool = True,
    attn_sink: torch.Tensor | None = None,
) -> dict:
    """Reference packed QK statistics. Input: [H, T, D].

    Slices each packed sequence and accumulates row-first statistics so the
    result matches ``compute_qk_stats_triton_packed`` exactly (mean is the
    average over valid rows of each row's mean logit — NOT a count-weighted
    position mean). This is the reference/fallback path; ``cu_seqlens.tolist()``
    incurs a D2H sync and is acceptable here (not the Triton hot path).

    ``attn_sink``: optional per-query-head sink logit ``[H]`` folded in as one
    extra key-less softmax column per sequence, matching the kernel.
    """
    num_heads, total_tokens, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)
    device = q.device

    bounds = cu_seqlens.to(torch.int64).tolist()

    sum_row_mean = torch.zeros(num_heads, device=device)
    sum_entropy = torch.zeros(num_heads, device=device)
    sum_sink = torch.zeros(num_heads, device=device)
    valid_rows = torch.zeros(num_heads, device=device)
    max_per_head = torch.full((num_heads,), -1e10, device=device)

    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        length = e - s
        if length <= 0:
            continue

        q_seq = q[:, s:e, :]  # [H, L, D]
        k_seq = k[:, s:e, :]
        logits = (
            torch.matmul(q_seq, k_seq.transpose(-2, -1)) * scale
        )  # [H, L, L]

        if causal:
            causal_mask = torch.triu(
                torch.ones(length, length, device=device, dtype=torch.bool),
                diagonal=1,
            )
            logits = logits.masked_fill(causal_mask, float("-inf"))

        valid_mask = logits > -1e9  # [H, L, L]
        row_count = valid_mask.sum(dim=-1)  # [H, L]

        # max over all valid positions in this sequence
        seq_max = torch.where(
            valid_mask, logits, torch.full_like(logits, -1e10)
        ).amax(dim=(-2, -1))
        max_per_head = torch.maximum(max_per_head, seq_max)

        # row-first mean
        row_sum = torch.where(valid_mask, logits, torch.zeros_like(logits)).sum(
            dim=-1
        )  # [H, L]
        row_mean = row_sum / row_count.clamp(min=1)  # [H, L]
        sum_row_mean += row_mean.sum(dim=-1)

        # entropy per row. Fold the sink logit in as an extra key-less column:
        # it enters the softmax denominator and entropy, but not max/mean or the
        # seq-start sink numerator (those describe real keys).
        if attn_sink is not None:
            sink_col = (
                attn_sink.reshape(num_heads, 1, 1)
                .expand(num_heads, length, 1)
                .to(logits.dtype)
            )
            ext_logits = torch.cat([logits, sink_col], dim=-1)
            ext_valid = torch.cat(
                [valid_mask, torch.ones_like(sink_col, dtype=torch.bool)],
                dim=-1,
            )
        else:
            ext_logits = logits
            ext_valid = valid_mask

        probs = torch.softmax(ext_logits, dim=-1)
        log_probs = torch.log_softmax(ext_logits, dim=-1)
        ent_map = -(probs * log_probs)
        ent_map = torch.where(ext_valid, ent_map, torch.zeros_like(ent_map))
        sum_entropy += ent_map.sum(dim=-1).sum(dim=-1)

        # sink = prob of each row's own sequence-start (local column 0)
        sum_sink += probs[..., 0].sum(dim=-1)

        valid_rows += (row_count > 0).sum(dim=-1).to(valid_rows.dtype)

    safe_rows = valid_rows.clamp(min=1)
    mean_per_head = (sum_row_mean / safe_rows).unsqueeze(0)
    entropy_per_head = (sum_entropy / safe_rows).unsqueeze(0)
    sink_per_head = (sum_sink / safe_rows).unsqueeze(0)
    max_per_head = max_per_head.unsqueeze(0)

    return {
        "max_per_head": max_per_head,
        "mean_per_head": mean_per_head,
        "entropy_per_head": entropy_per_head,
        "sink_per_head": sink_per_head,
        "max_global": max_per_head.max(),
        "mean_global": mean_per_head.mean(),
        "entropy_global": entropy_per_head.mean(),
        "sink_global": sink_per_head.mean(),
    }


def compute_qk_stats_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    causal: bool = True,
    use_triton: bool = True,
    attn_sink: torch.Tensor | None = None,
) -> dict:
    """Unified packed (THD) entry point. Input layout: [T, H, D].

    Respects per-sequence boundaries from ``cu_seqlens`` so no attention
    leaks across packed sequences and each sequence's sink is its own first
    token. GQA is expanded on the host (matching ``compute_qk_stats``), then
    the tensors are permuted to [H, T, D] before dispatch.

    ``attn_sink``: optional per-query-head sink logit ``[H]`` (softmax_offset).
    """
    total_tokens, num_q_heads, head_dim = q.shape
    _, num_k_heads, _ = k.shape

    if num_q_heads != num_k_heads:
        heads_per_group = num_q_heads // num_k_heads
        k = k.repeat_interleave(heads_per_group, dim=1)

    # [T, H, D] -> [H, T, D]
    q = q.permute(1, 0, 2).contiguous()
    k = k.permute(1, 0, 2).contiguous()

    if use_triton:
        return compute_qk_stats_triton_packed(
            q, k, cu_seqlens, causal, attn_sink=attn_sink
        )

    return compute_qk_stats_pytorch_packed(
        q, k, cu_seqlens, causal, attn_sink=attn_sink
    )
