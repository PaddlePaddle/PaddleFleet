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

"""Shared Triton kernel for QK attention statistics — framework-independent.

Computes per-head: max logit, mean logit, entropy, sink weight
using online softmax (O(S) memory, no S×S materialization).

Supports asymmetric Q/K sequence lengths for Context-Parallel (CP) monitoring:
each CP rank keeps its local Q shard (``seq_len_q = S/CP``) but sees the
full-sequence K (``seq_len_k = S``) after an all_gather. Passing
``q_row_offset = cp_rank * seq_len_q`` restores correct causal masking.

Three families of kernels live here:

* ``qk_stats_kernel`` / ``qk_stats_partial_kernel`` — dense causal attention.
* ``qk_stats_sparse_partial_kernel`` — two-segment sparse attention, used for
  stacks whose per-layer kind comes from ``csa_compress_ratios``, where a query
  attends to a sliding window over the original KV plus a contiguous run of
  compressed KV entries, under a single softmax that also includes a learned
  per-head sink logit.
* ``qk_stats_packed_kernel`` — packed (THD) variant of the partial kernel: when
  multiple variable-length sequences are packed into one ``[H, T, D]`` tensor,
  per-token ``seq_start``/``seq_end`` boundaries confine each query to its own
  sequence (no cross-sequence leakage) and the sink column is each row's own
  sequence start rather than global token 0.
"""

import triton
import triton.language as tl


@triton.jit
def qk_stats_kernel(
    # Pointers
    Q_ptr,
    K_ptr,
    attn_sink_ptr,
    max_logits_ptr,
    mean_logits_ptr,
    entropy_ptr,
    sink_ptr,
    count_ptr,
    # Shapes
    batch,
    num_heads,
    seq_len_q,
    seq_len_k,
    head_dim,
    # GQA: number of query heads sharing one KV head (1 == MHA)
    heads_per_group,
    # Query row offset in the global sequence (0 unless CP > 1)
    q_row_offset,
    # Strides
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    stride_out_batch,
    stride_out_head,
    # Configs
    scale: tl.constexpr,
    apply_causal_mask: tl.constexpr,
    HAS_SINK: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused kernel for QK attention statistics via online softmax.

    Grid: (batch * num_heads,) — one program per (batch, query-head).
    Input layout: [B, H, S, D].

    Asymmetric Q/K seq lens (for CP): ``seq_len_q`` may be smaller than
    ``seq_len_k``. Row m in Q corresponds to *global* row ``m + q_row_offset``
    which is used only for causal masking against K columns.

    GQA: K is indexed by ``head_idx // heads_per_group`` so the caller does
    NOT need to materialize a ``repeat_interleave`` of the KV tensor. Pass
    ``heads_per_group=1`` for MHA (no grouping).

    Query-row subsampling: only every ``ROW_STRIDE``-th query row is visited
    in the outer loop. The mean-class statistics (mean_logit, entropy, sink)
    are row-averages, so a uniform stride is an unbiased estimator of the
    full-sequence average at O(S/ROW_STRIDE) cost instead of O(S). Pass
    ``ROW_STRIDE=1`` to recover the exact full-sequence behavior. Note that
    ``max_logit`` is an extremum over the visited rows; with ROW_STRIDE>1 it
    is a (typically tight) lower bound on the true max.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads
    kv_head_idx = head_idx // heads_per_group

    q_base = Q_ptr + batch_idx * stride_q_batch + head_idx * stride_q_head
    k_base = K_ptr + batch_idx * stride_k_batch + kv_head_idx * stride_k_head

    head_max_logit = -1e10
    head_sum_logit = 0.0
    head_sum_row_mean = 0.0  # sum over valid rows of (per-row mean logit)
    head_valid_count = 0.0
    head_sum_entropy = 0.0
    head_sum_sink = 0.0
    head_valid_rows = 0.0

    # Stride between consecutive visited query rows within a block.
    m_block_span = BLOCK_M * ROW_STRIDE
    for m_start in range(0, seq_len_q, m_block_span):
        m_offsets = m_start + tl.arange(0, BLOCK_M) * ROW_STRIDE
        m_mask = m_offsets < seq_len_q
        # Global row index used for causal masking against K columns
        m_global = m_offsets + q_row_offset

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
        d_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        h_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        s_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        row_max_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
        row_sum_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32)
        row_count_raw = tl.zeros([BLOCK_M], dtype=tl.float32)

        if apply_causal_mask:
            n_stop = tl.minimum(
                q_row_offset + m_start + (BLOCK_M - 1) * ROW_STRIDE + BLOCK_N,
                seq_len_k,
            )
        else:
            n_stop = seq_len_k

        for n_start in range(0, n_stop, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offsets < seq_len_k

            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for k_start in range(0, head_dim, BLOCK_K):
                k_offsets = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offsets < head_dim

                q_ptr = (
                    q_base
                    + m_offsets[:, None] * stride_q_seq
                    + k_offsets[None, :] * stride_q_dim
                )
                q = tl.load(
                    q_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0
                )

                k_ptr = (
                    k_base
                    + n_offsets[:, None] * stride_k_seq
                    + k_offsets[None, :] * stride_k_dim
                )
                k = tl.load(
                    k_ptr, mask=n_mask[:, None] & k_mask[None, :], other=0.0
                )

                acc += tl.dot(q, tl.trans(k))

            logits = acc * scale

            mask_val = n_mask[None, :] & m_mask[:, None]
            if apply_causal_mask:
                causal = m_global[:, None] >= n_offsets[None, :]
                mask_val = mask_val & causal

            logits = tl.where(mask_val, logits, -1e10)

            block_max = tl.max(logits, 1)
            row_max_logit_raw = tl.maximum(row_max_logit_raw, block_max)

            logits_zeroed = tl.where(mask_val, logits, 0.0)
            row_sum_logit_raw += tl.sum(logits_zeroed, 1)
            row_count_raw += tl.sum(mask_val.to(tl.float32), 1)

            # Online softmax update
            m_curr = block_max
            m_new = tl.maximum(m_i, m_curr)

            alpha_prev = tl.exp(m_i - m_new)
            alpha_curr = tl.exp(logits - m_new[:, None])
            alpha_curr = tl.where(mask_val, alpha_curr, 0.0)

            d_curr = tl.sum(alpha_curr, 1)
            d_new = d_i * alpha_prev + d_curr

            h_term_prev = h_i * alpha_prev + (m_i - m_new) * d_i * alpha_prev
            logits_diff = logits - m_new[:, None]
            h_term_curr = tl.sum(
                tl.where(mask_val, logits_diff * alpha_curr, 0.0), 1
            )
            h_new = h_term_prev + h_term_curr

            if n_start == 0:
                is_first = n_offsets == 0
                sink_contrib = tl.sum(
                    tl.where(is_first[None, :], alpha_curr, 0.0), 1
                )
                s_new = s_i * alpha_prev + sink_contrib
            else:
                s_new = s_i * alpha_prev

            m_i = m_new
            d_i = d_new
            h_i = h_new
            s_i = s_new

        # Fold the per-head sink logit as an extra key-less column: it enters
        # the denominator (and entropy) but not max/mean or the sink numerator.
        # h uses the OLD d_i before d_i is rescaled (see partial kernel).
        if HAS_SINK:
            sink_logit = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
            m_new = tl.maximum(m_i, sink_logit)
            alpha_prev = tl.exp(m_i - m_new)
            h_i = (h_i + (m_i - m_new) * d_i) * alpha_prev
            d_i = d_i * alpha_prev
            s_i = s_i * alpha_prev
            sink_w = tl.exp(sink_logit - m_new)
            d_i = d_i + sink_w
            h_i = h_i + sink_w * (sink_logit - m_new)
            m_i = m_new

        # Finalize block rows
        log_d = tl.log(d_i)
        row_entropy = log_d - (h_i / d_i)
        row_sink = s_i / d_i

        row_has_data = row_count_raw > 0

        head_max_logit = tl.maximum(head_max_logit, tl.max(row_max_logit_raw))
        head_sum_logit += tl.sum(tl.where(row_has_data, row_sum_logit_raw, 0.0))
        head_valid_count += tl.sum(row_count_raw)

        # Per-row mean logit (accumulated across M-blocks for row-first mean).
        safe_row_count = tl.where(row_count_raw > 0, row_count_raw, 1.0)
        row_mean_logit = row_sum_logit_raw / safe_row_count
        head_sum_row_mean += tl.sum(
            tl.where(row_has_data & m_mask, row_mean_logit, 0.0)
        )

        head_sum_entropy += tl.sum(
            tl.where(row_has_data & m_mask, row_entropy, 0.0)
        )
        head_sum_sink += tl.sum(tl.where(row_has_data & m_mask, row_sink, 0.0))
        head_valid_rows += tl.sum((row_has_data & m_mask).to(tl.float32))

    # Write output
    out_offset = batch_idx * stride_out_batch + head_idx * stride_out_head

    # NOTE: mean_logit uses row-first semantics (mean over valid rows of the
    # per-row mean logit) rather than count-weighted position mean.
    safe_rows = tl.maximum(head_valid_rows, 1.0)

    tl.store(max_logits_ptr + out_offset, head_max_logit)
    tl.store(mean_logits_ptr + out_offset, head_sum_row_mean / safe_rows)
    tl.store(entropy_ptr + out_offset, head_sum_entropy / safe_rows)
    tl.store(sink_ptr + out_offset, head_sum_sink / safe_rows)
    tl.store(count_ptr + out_offset, head_valid_count)


@triton.jit
def qk_stats_partial_kernel(
    # Pointers
    Q_ptr,
    K_ptr,
    attn_sink_ptr,
    partial_max_ptr,
    partial_sum_logit_ptr,
    partial_sum_row_mean_ptr,
    partial_count_ptr,
    partial_sum_entropy_ptr,
    partial_sum_sink_ptr,
    partial_valid_rows_ptr,
    # Shapes
    batch,
    num_heads,
    seq_len_q,
    seq_len_k,
    head_dim,
    heads_per_group,
    num_m_blocks,
    # Query row offset in the global sequence (0 unless CP > 1)
    q_row_offset,
    # Strides for Q/K (input layout [B, H, S, D])
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    # Strides for partial buffers (layout [B, H, num_m_blocks])
    stride_p_batch,
    stride_p_head,
    stride_p_m,
    # Configs
    scale: tl.constexpr,
    apply_causal_mask: tl.constexpr,
    HAS_SINK: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Grid: (batch * num_heads, num_m_blocks). Each program handles exactly one
    M-block, replacing the outer ``for m_start in range(...)`` loop of the
    legacy single-program kernel.

    Asymmetric Q/K seq lens (for CP): ``seq_len_q`` may be smaller than
    ``seq_len_k``. Row m in Q corresponds to *global* row ``m + q_row_offset``
    which is used only for causal masking against K columns.

    ``HAS_SINK``: fold a learned per-head sink logit (``attn_sink_ptr``, one
    scalar per query head) into the same softmax. Full-attention layers carry it
    as ``core_attention.softmax_offset`` and hand it to their kernel as
    ``learnable_sink``; ``off-by-one`` softmax is the same thing with the logit
    pinned to 0. Without this the reported distribution is renormalised over the
    real keys only, which is not the distribution the model computes.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    kv_head_idx = head_idx // heads_per_group

    q_base = Q_ptr + batch_idx * stride_q_batch + head_idx * stride_q_head
    k_base = K_ptr + batch_idx * stride_k_batch + kv_head_idx * stride_k_head

    m_block_span = BLOCK_M * ROW_STRIDE
    m_start = pid_m * m_block_span

    m_offsets = m_start + tl.arange(0, BLOCK_M) * ROW_STRIDE
    m_mask = m_offsets < seq_len_q
    # Global row index used for causal masking against K columns
    m_global = m_offsets + q_row_offset

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    h_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    s_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    row_max_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    row_sum_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32)
    row_count_raw = tl.zeros([BLOCK_M], dtype=tl.float32)

    if apply_causal_mask:
        n_stop = tl.minimum(
            q_row_offset + m_start + (BLOCK_M - 1) * ROW_STRIDE + BLOCK_N,
            seq_len_k,
        )
    else:
        n_stop = seq_len_k

    for n_start in range(0, n_stop, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < seq_len_k

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in range(0, head_dim, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < head_dim

            q_ptr = (
                q_base
                + m_offsets[:, None] * stride_q_seq
                + k_offsets[None, :] * stride_q_dim
            )
            q = tl.load(
                q_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0
            )

            k_ptr = (
                k_base
                + n_offsets[:, None] * stride_k_seq
                + k_offsets[None, :] * stride_k_dim
            )
            k = tl.load(
                k_ptr, mask=n_mask[:, None] & k_mask[None, :], other=0.0
            )

            acc += tl.dot(q, tl.trans(k))

        logits = acc * scale

        mask_val = n_mask[None, :] & m_mask[:, None]
        if apply_causal_mask:
            causal = m_global[:, None] >= n_offsets[None, :]
            mask_val = mask_val & causal

        logits = tl.where(mask_val, logits, -1e10)

        block_max = tl.max(logits, 1)
        row_max_logit_raw = tl.maximum(row_max_logit_raw, block_max)

        logits_zeroed = tl.where(mask_val, logits, 0.0)
        row_sum_logit_raw += tl.sum(logits_zeroed, 1)
        row_count_raw += tl.sum(mask_val.to(tl.float32), 1)

        # Online softmax update
        m_curr = block_max
        m_new = tl.maximum(m_i, m_curr)

        alpha_prev = tl.exp(m_i - m_new)
        alpha_curr = tl.exp(logits - m_new[:, None])
        alpha_curr = tl.where(mask_val, alpha_curr, 0.0)

        d_curr = tl.sum(alpha_curr, 1)
        d_new = d_i * alpha_prev + d_curr

        h_term_prev = h_i * alpha_prev + (m_i - m_new) * d_i * alpha_prev
        logits_diff = logits - m_new[:, None]
        h_term_curr = tl.sum(
            tl.where(mask_val, logits_diff * alpha_curr, 0.0), 1
        )
        h_new = h_term_prev + h_term_curr

        if n_start == 0:
            is_first = n_offsets == 0
            sink_contrib = tl.sum(
                tl.where(is_first[None, :], alpha_curr, 0.0), 1
            )
            s_new = s_i * alpha_prev + sink_contrib
        else:
            s_new = s_i * alpha_prev

        m_i = m_new
        d_i = d_new
        h_i = h_new
        s_i = s_new

    # Fold the sink logit into the same softmax. It is an extra key-less column,
    # so it changes the denominator (and therefore entropy and sink) but is
    # deliberately excluded from logit max/mean, which describe real keys. Same
    # rescale as the in-loop update: shifting the max by (m_i - m_new) moves h by
    # (m_i - m_new) * d, so h must be updated from the old d.
    if HAS_SINK:
        sink_logit = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
        m_new = tl.maximum(m_i, sink_logit)
        alpha_prev = tl.exp(m_i - m_new)
        h_i = (h_i + (m_i - m_new) * d_i) * alpha_prev
        d_i = d_i * alpha_prev
        s_i = s_i * alpha_prev
        sink_w = tl.exp(sink_logit - m_new)
        d_i = d_i + sink_w
        h_i = h_i + sink_w * (sink_logit - m_new)
        m_i = m_new

    # Finalize per-row stats for this M-block
    safe_d = tl.where(d_i > 0, d_i, 1.0)
    row_entropy = tl.log(safe_d) - (h_i / safe_d)
    row_sink = s_i / safe_d

    row_has_data = row_count_raw > 0
    valid_mask = row_has_data & m_mask

    block_max_logit = tl.max(tl.where(m_mask, row_max_logit_raw, -1e10))
    block_sum_logit = tl.sum(tl.where(valid_mask, row_sum_logit_raw, 0.0))
    # Row-first mean: sum of per-row means (needed so CP composes via mean-of-means).
    safe_row_count = tl.where(row_count_raw > 0, row_count_raw, 1.0)
    row_mean_logit = row_sum_logit_raw / safe_row_count
    block_sum_row_mean = tl.sum(tl.where(valid_mask, row_mean_logit, 0.0))
    block_count = tl.sum(tl.where(m_mask, row_count_raw, 0.0))
    block_sum_entropy = tl.sum(tl.where(valid_mask, row_entropy, 0.0))
    block_sum_sink = tl.sum(tl.where(valid_mask, row_sink, 0.0))
    block_valid_rows = tl.sum(valid_mask.to(tl.float32))

    out_offset = (
        batch_idx * stride_p_batch
        + head_idx * stride_p_head
        + pid_m * stride_p_m
    )

    tl.store(partial_max_ptr + out_offset, block_max_logit)
    tl.store(partial_sum_logit_ptr + out_offset, block_sum_logit)
    tl.store(partial_sum_row_mean_ptr + out_offset, block_sum_row_mean)
    tl.store(partial_count_ptr + out_offset, block_count)
    tl.store(partial_sum_entropy_ptr + out_offset, block_sum_entropy)
    tl.store(partial_sum_sink_ptr + out_offset, block_sum_sink)
    tl.store(partial_valid_rows_ptr + out_offset, block_valid_rows)


@triton.jit
def _accumulate_segment(
    q_base,
    k_base,
    m_offsets,
    m_mask,
    row_lo,
    row_hi,
    sink_col,
    seg_start,
    seg_stop,
    seq_len_k,
    head_dim,
    stride_q_seq,
    stride_q_dim,
    stride_k_seq,
    stride_k_dim,
    m_i,
    d_i,
    h_i,
    s_i,
    row_max_logit_raw,
    row_sum_logit_raw,
    row_count_raw,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fold one contiguous key segment into the running online-softmax state.

    ``row_lo`` / ``row_hi`` are per-row *inclusive* key bounds within this
    segment; a row with ``row_hi < row_lo`` contributes nothing. ``seg_start`` /
    ``seg_stop`` are the scalar loop bounds covering the union of the block's
    rows, so the visited column span stays proportional to the segment width
    instead of the full sequence.

    Calling this twice (window segment, then compressed segment) yields exactly
    the same result as one softmax over the union of both key sets, because the
    online-softmax state composes across disjoint column ranges.
    """
    for n_start in range(seg_start, seg_stop, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < seq_len_k

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for k_start in range(0, head_dim, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < head_dim

            q_ptr = (
                q_base
                + m_offsets[:, None] * stride_q_seq
                + k_offsets[None, :] * stride_q_dim
            )
            q = tl.load(
                q_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0
            )

            k_ptr = (
                k_base
                + n_offsets[:, None] * stride_k_seq
                + k_offsets[None, :] * stride_k_dim
            )
            k = tl.load(
                k_ptr, mask=n_mask[:, None] & k_mask[None, :], other=0.0
            )

            acc += tl.dot(q, tl.trans(k))

        logits = acc * scale

        in_range = (n_offsets[None, :] >= row_lo[:, None]) & (
            n_offsets[None, :] <= row_hi[:, None]
        )
        mask_val = n_mask[None, :] & m_mask[:, None] & in_range

        logits = tl.where(mask_val, logits, -1e10)

        block_max = tl.max(logits, 1)
        row_max_logit_raw = tl.maximum(row_max_logit_raw, block_max)

        logits_zeroed = tl.where(mask_val, logits, 0.0)
        row_sum_logit_raw += tl.sum(logits_zeroed, 1)
        row_count_raw += tl.sum(mask_val.to(tl.float32), 1)

        # Online softmax update
        m_new = tl.maximum(m_i, block_max)

        alpha_prev = tl.exp(m_i - m_new)
        alpha_curr = tl.exp(logits - m_new[:, None])
        alpha_curr = tl.where(mask_val, alpha_curr, 0.0)

        # h tracks sum of p_unnorm * (logit - m); rescaling it on an m shift
        # needs the *previous* d, so update h before d.
        h_term_prev = h_i * alpha_prev + (m_i - m_new) * d_i * alpha_prev
        logits_diff = logits - m_new[:, None]
        h_i = h_term_prev + tl.sum(
            tl.where(mask_val, logits_diff * alpha_curr, 0.0), 1
        )

        d_i = d_i * alpha_prev + tl.sum(alpha_curr, 1)

        # Sink weight follows the row's earliest reachable key (column 0 for
        # dense causal attention, the first compressed block otherwise), so it
        # can be picked up from any segment, not just the first block.
        is_sink = n_offsets[None, :] == sink_col[:, None]
        sink_contrib = tl.sum(tl.where(is_sink & mask_val, alpha_curr, 0.0), 1)
        s_i = s_i * alpha_prev + sink_contrib

        m_i = m_new

    return (
        m_i,
        d_i,
        h_i,
        s_i,
        row_max_logit_raw,
        row_sum_logit_raw,
        row_count_raw,
    )


@triton.jit
def qk_stats_sparse_partial_kernel(
    # Pointers
    Q_ptr,
    K_ptr,
    win_lo_ptr,
    win_hi_ptr,
    cmp_lo_ptr,
    cmp_hi_ptr,
    sink_col_ptr,
    attn_sink_ptr,
    partial_max_ptr,
    partial_sum_logit_ptr,
    partial_sum_row_mean_ptr,
    partial_count_ptr,
    partial_sum_entropy_ptr,
    partial_sum_sink_ptr,
    partial_valid_rows_ptr,
    # Shapes
    batch,
    num_heads,
    seq_len_q,
    seq_len_k,
    head_dim,
    heads_per_group,
    num_m_blocks,
    # Strides for Q/K (input layout [B, H, S, D])
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    # Strides for the per-row bound tensors (layout [B, S_q])
    stride_b_batch,
    stride_b_seq,
    # Strides for partial buffers (layout [B, H, num_m_blocks])
    stride_p_batch,
    stride_p_head,
    stride_p_m,
    # Configs
    scale: tl.constexpr,
    HAS_SINK: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """QK statistics over a two-segment sparse key set (window + compressed KV).

    Grid: ``(batch * num_heads, num_m_blocks)`` — same split-M layout as
    :func:`qk_stats_partial_kernel`, and it writes the same partial buffers so
    the host-side reduction is shared.

    The key set of query row ``g`` is ``[win_lo[g], win_hi[g]]`` over the
    original KV plus ``[cmp_lo[g], cmp_hi[g]]`` over the compressed KV, both
    inclusive, both indexing the concatenated ``kv_full``. An empty segment is
    encoded as ``hi < lo``. These bounds come from the very same index helpers
    the model uses, so document-boundary resets are honoured.

    ``attn_sink_ptr`` holds the learned per-head sink logit that the model adds
    to the softmax denominator; it is folded in after both segments so entropy
    and sink ratios match the distribution the layer actually computes.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    kv_head_idx = head_idx // heads_per_group

    q_base = Q_ptr + batch_idx * stride_q_batch + head_idx * stride_q_head
    k_base = K_ptr + batch_idx * stride_k_batch + kv_head_idx * stride_k_head

    m_block_span = BLOCK_M * ROW_STRIDE
    m_start = pid_m * m_block_span

    m_offsets = m_start + tl.arange(0, BLOCK_M) * ROW_STRIDE
    m_mask = m_offsets < seq_len_q

    bound_ptr = (
        win_lo_ptr + batch_idx * stride_b_batch + m_offsets * stride_b_seq
    )
    win_lo = tl.load(bound_ptr, mask=m_mask, other=0)
    win_hi = tl.load(
        win_hi_ptr + batch_idx * stride_b_batch + m_offsets * stride_b_seq,
        mask=m_mask,
        other=-1,
    )
    cmp_lo = tl.load(
        cmp_lo_ptr + batch_idx * stride_b_batch + m_offsets * stride_b_seq,
        mask=m_mask,
        other=0,
    )
    cmp_hi = tl.load(
        cmp_hi_ptr + batch_idx * stride_b_batch + m_offsets * stride_b_seq,
        mask=m_mask,
        other=-1,
    )
    sink_col = tl.load(
        sink_col_ptr + batch_idx * stride_b_batch + m_offsets * stride_b_seq,
        mask=m_mask,
        other=-1,
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    h_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    s_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    row_max_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    row_sum_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32)
    row_count_raw = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Window segment
    win_start = tl.min(tl.where(m_mask, win_lo, seq_len_k))
    win_stop = tl.max(tl.where(m_mask, win_hi, -1)) + 1
    m_i, d_i, h_i, s_i, row_max_logit_raw, row_sum_logit_raw, row_count_raw = (
        _accumulate_segment(
            q_base,
            k_base,
            m_offsets,
            m_mask,
            win_lo,
            win_hi,
            sink_col,
            win_start,
            win_stop,
            seq_len_k,
            head_dim,
            stride_q_seq,
            stride_q_dim,
            stride_k_seq,
            stride_k_dim,
            m_i,
            d_i,
            h_i,
            s_i,
            row_max_logit_raw,
            row_sum_logit_raw,
            row_count_raw,
            scale=scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    )

    # Compressed segment
    cmp_start = tl.min(tl.where(m_mask, cmp_lo, seq_len_k))
    cmp_stop = tl.max(tl.where(m_mask, cmp_hi, -1)) + 1
    m_i, d_i, h_i, s_i, row_max_logit_raw, row_sum_logit_raw, row_count_raw = (
        _accumulate_segment(
            q_base,
            k_base,
            m_offsets,
            m_mask,
            cmp_lo,
            cmp_hi,
            sink_col,
            cmp_start,
            cmp_stop,
            seq_len_k,
            head_dim,
            stride_q_seq,
            stride_q_dim,
            stride_k_seq,
            stride_k_dim,
            m_i,
            d_i,
            h_i,
            s_i,
            row_max_logit_raw,
            row_sum_logit_raw,
            row_count_raw,
            scale=scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
    )

    # Fold the learned sink logit into the same softmax. The sink is an extra
    # (key-less) column, so it changes the denominator and the entropy but is
    # deliberately excluded from logit max/mean, which describe real keys.
    if HAS_SINK:
        sink_logit = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
        m_new = tl.maximum(m_i, sink_logit)
        alpha_prev = tl.exp(m_i - m_new)
        # Same rescale as the in-loop update: shifting the max by (m_i - m_new)
        # moves h by (m_i - m_new) * d, so h must be updated from the old d.
        h_i = (h_i + (m_i - m_new) * d_i) * alpha_prev
        d_i = d_i * alpha_prev
        s_i = s_i * alpha_prev
        sink_w = tl.exp(sink_logit - m_new)
        d_i = d_i + sink_w
        h_i = h_i + sink_w * (sink_logit - m_new)
        m_i = m_new

    # Finalize per-row stats for this M-block
    safe_d = tl.where(d_i > 0, d_i, 1.0)
    row_entropy = tl.log(safe_d) - (h_i / safe_d)
    row_sink = s_i / safe_d

    row_has_data = row_count_raw > 0
    valid_mask = row_has_data & m_mask

    block_max_logit = tl.max(tl.where(m_mask, row_max_logit_raw, -1e10))
    block_sum_logit = tl.sum(tl.where(valid_mask, row_sum_logit_raw, 0.0))
    safe_row_count = tl.where(row_count_raw > 0, row_count_raw, 1.0)
    row_mean_logit = row_sum_logit_raw / safe_row_count
    block_sum_row_mean = tl.sum(tl.where(valid_mask, row_mean_logit, 0.0))
    block_count = tl.sum(tl.where(m_mask, row_count_raw, 0.0))
    block_sum_entropy = tl.sum(tl.where(valid_mask, row_entropy, 0.0))
    block_sum_sink = tl.sum(tl.where(valid_mask, row_sink, 0.0))
    block_valid_rows = tl.sum(valid_mask.to(tl.float32))

    out_offset = (
        batch_idx * stride_p_batch
        + head_idx * stride_p_head
        + pid_m * stride_p_m
    )

    tl.store(partial_max_ptr + out_offset, block_max_logit)
    tl.store(partial_sum_logit_ptr + out_offset, block_sum_logit)
    tl.store(partial_sum_row_mean_ptr + out_offset, block_sum_row_mean)
    tl.store(partial_count_ptr + out_offset, block_count)
    tl.store(partial_sum_entropy_ptr + out_offset, block_sum_entropy)
    tl.store(partial_sum_sink_ptr + out_offset, block_sum_sink)
    tl.store(partial_valid_rows_ptr + out_offset, block_valid_rows)


@triton.jit
def qk_stats_packed_kernel(
    # Pointers
    Q_ptr,
    K_ptr,
    attn_sink_ptr,
    token_seq_start_ptr,  # [total_tokens] int32: start token index of each token's sequence
    token_seq_end_ptr,  # [total_tokens] int32: end (exclusive) token index of each token's sequence
    partial_max_ptr,
    partial_sum_logit_ptr,
    partial_sum_row_mean_ptr,
    partial_count_ptr,
    partial_sum_entropy_ptr,
    partial_sum_sink_ptr,
    partial_valid_rows_ptr,
    # Shapes
    num_heads,
    total_tokens,
    head_dim,
    num_m_blocks,
    # Strides for Q/K (THD layout [H, T, D]; batch is always 1 for packed)
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    # Strides for partial buffers (layout [H, num_m_blocks])
    stride_p_head,
    stride_p_m,
    # Configs
    scale: tl.constexpr,
    apply_causal_mask: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Packed (THD) variant of ``qk_stats_partial_kernel``.

    Grid: (num_heads, num_m_blocks). Input layout: [H, T, D] where T is the
    total packed-token count across all sequences (batch dim is always 1).

    Per-sequence semantics via ``token_seq_start``/``token_seq_end`` (built on
    the host from ``cu_seqlens`` — see ``_cu_seqlens_to_token_arrays``):
      * a query row m only attends to key columns n with
        ``seq_start[m] <= n < seq_end[m]`` (no cross-sequence leakage), and
      * the attention-sink column for row m is that row's own sequence start
        ``seq_start[m]`` — NOT global token 0.
    Padding tokens carry ``seq_start == seq_end == 0`` so ``same_seq`` masks
    them out (empty valid range) and they contribute nothing.

    NOTE: CP (``q_row_offset`` / asymmetric seq lens) and ``ROW_STRIDE``
    subsampling are intentionally unsupported here — packed monitoring does not
    yet compose with either. Add them only alongside a correctness test.
    """
    pid_head = tl.program_id(0)
    pid_m = tl.program_id(1)

    head_idx = pid_head
    q_base = Q_ptr + head_idx * stride_q_head
    # GQA is expanded on the host (repeat_interleave), so K shares Q's head idx.
    k_base = K_ptr + head_idx * stride_k_head

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < total_tokens

    # Per-row sequence boundaries (int32). Padding rows -> [0, 0) empty range.
    seq_start = tl.load(token_seq_start_ptr + m_offsets, mask=m_mask, other=0)
    seq_end = tl.load(token_seq_end_ptr + m_offsets, mask=m_mask, other=0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    h_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    s_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    row_max_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    row_sum_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32)
    row_count_raw = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Bound the key scan to the columns any row in this block can actually see:
    # lower = smallest sequence start among valid rows (skip fully-masked
    # earlier sequences); upper = causal cap (n <= max m) or max sequence end.
    n_lo = tl.min(tl.where(m_mask, seq_start, total_tokens))
    n_start_init = (n_lo // BLOCK_N) * BLOCK_N
    if apply_causal_mask:
        n_stop = tl.minimum(m_start + BLOCK_M, total_tokens)
    else:
        n_stop = tl.minimum(tl.max(tl.where(m_mask, seq_end, 0)), total_tokens)

    for n_start in range(n_start_init, n_stop, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < total_tokens

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in range(0, head_dim, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < head_dim

            q_ptr = (
                q_base
                + m_offsets[:, None] * stride_q_seq
                + k_offsets[None, :] * stride_q_dim
            )
            q = tl.load(
                q_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0
            )

            k_ptr = (
                k_base
                + n_offsets[:, None] * stride_k_seq
                + k_offsets[None, :] * stride_k_dim
            )
            k = tl.load(
                k_ptr, mask=n_mask[:, None] & k_mask[None, :], other=0.0
            )

            acc += tl.dot(q, tl.trans(k))

        logits = acc * scale

        # Same-sequence mask: n in [seq_start[m], seq_end[m]). Global offsets are
        # used directly — same_seq already prevents any cross-sequence leakage.
        same_seq = (n_offsets[None, :] >= seq_start[:, None]) & (
            n_offsets[None, :] < seq_end[:, None]
        )
        mask_val = n_mask[None, :] & m_mask[:, None] & same_seq
        if apply_causal_mask:
            causal = m_offsets[:, None] >= n_offsets[None, :]
            mask_val = mask_val & causal

        logits = tl.where(mask_val, logits, -1e10)

        block_max = tl.max(logits, 1)
        row_max_logit_raw = tl.maximum(row_max_logit_raw, block_max)

        logits_zeroed = tl.where(mask_val, logits, 0.0)
        row_sum_logit_raw += tl.sum(logits_zeroed, 1)
        row_count_raw += tl.sum(mask_val.to(tl.float32), 1)

        # Online softmax update
        m_curr = block_max
        m_new = tl.maximum(m_i, m_curr)

        alpha_prev = tl.exp(m_i - m_new)
        alpha_curr = tl.exp(logits - m_new[:, None])
        alpha_curr = tl.where(mask_val, alpha_curr, 0.0)

        d_curr = tl.sum(alpha_curr, 1)
        d_new = d_i * alpha_prev + d_curr

        h_term_prev = h_i * alpha_prev + (m_i - m_new) * d_i * alpha_prev
        logits_diff = logits - m_new[:, None]
        h_term_curr = tl.sum(
            tl.where(mask_val, logits_diff * alpha_curr, 0.0), 1
        )
        h_new = h_term_prev + h_term_curr

        # Sink = per-row sequence start column. Unlike the dense kernel this is
        # not confined to n_start==0, since each row's sink lives at seq_start[m]
        # which can fall in any key block.
        is_sink_col = n_offsets[None, :] == seq_start[:, None]
        sink_contrib = tl.sum(
            tl.where(mask_val & is_sink_col, alpha_curr, 0.0), 1
        )
        s_new = s_i * alpha_prev + sink_contrib

        m_i = m_new
        d_i = d_new
        h_i = h_new
        s_i = s_new

    # Fold the per-head sink logit as an extra key-less column on top of the
    # per-sequence sink column: it enters the denominator (and entropy) but not
    # max/mean or the sink numerator. h uses the OLD d_i before d_i is rescaled.
    if HAS_SINK:
        sink_logit = tl.load(attn_sink_ptr + head_idx).to(tl.float32)
        m_new = tl.maximum(m_i, sink_logit)
        alpha_prev = tl.exp(m_i - m_new)
        h_i = (h_i + (m_i - m_new) * d_i) * alpha_prev
        d_i = d_i * alpha_prev
        s_i = s_i * alpha_prev
        sink_w = tl.exp(sink_logit - m_new)
        d_i = d_i + sink_w
        h_i = h_i + sink_w * (sink_logit - m_new)
        m_i = m_new

    # Finalize per-row stats for this M-block
    safe_d = tl.where(d_i > 0, d_i, 1.0)
    row_entropy = tl.log(safe_d) - (h_i / safe_d)
    row_sink = s_i / safe_d

    row_has_data = row_count_raw > 0
    valid_mask = row_has_data & m_mask

    block_max_logit = tl.max(tl.where(m_mask, row_max_logit_raw, -1e10))
    block_sum_logit = tl.sum(tl.where(valid_mask, row_sum_logit_raw, 0.0))
    safe_row_count = tl.where(row_count_raw > 0, row_count_raw, 1.0)
    row_mean_logit = row_sum_logit_raw / safe_row_count
    block_sum_row_mean = tl.sum(tl.where(valid_mask, row_mean_logit, 0.0))
    block_count = tl.sum(tl.where(m_mask, row_count_raw, 0.0))
    block_sum_entropy = tl.sum(tl.where(valid_mask, row_entropy, 0.0))
    block_sum_sink = tl.sum(tl.where(valid_mask, row_sink, 0.0))
    block_valid_rows = tl.sum(valid_mask.to(tl.float32))

    out_offset = head_idx * stride_p_head + pid_m * stride_p_m

    tl.store(partial_max_ptr + out_offset, block_max_logit)
    tl.store(partial_sum_logit_ptr + out_offset, block_sum_logit)
    tl.store(partial_sum_row_mean_ptr + out_offset, block_sum_row_mean)
    tl.store(partial_count_ptr + out_offset, block_count)
    tl.store(partial_sum_entropy_ptr + out_offset, block_sum_entropy)
    tl.store(partial_sum_sink_ptr + out_offset, block_sum_sink)
    tl.store(partial_valid_rows_ptr + out_offset, block_valid_rows)
