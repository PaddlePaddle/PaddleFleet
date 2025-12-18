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

import math

import paddle

paddle.compat.enable_torch_proxy()

import triton
import triton.language as tl

from .block_mask_utils import (
    check_fully_masked_state,
    check_partially_masked_state,
)
from .index_utils import (
    prepare_maxmin,
)


@triton.jit
def flashmask_apply(
    X,
    q_rows,
    base_offset,
    k_offsets,
    load_mask,
    lt_start_ptr,
    lt_end_ptr,
    ut_start_ptr,
    ut_end_ptr,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    INT_MAX: tl.constexpr = 2147483647
    INT_MIN: tl.constexpr = -2147483648

    pad_lt = INT_MAX
    pad_ut = INT_MIN

    lts = tl.load(
        lt_start_ptr + base_offset + k_offsets, mask=load_mask, other=pad_lt
    )
    if mode == 1:
        dense_mask = q_rows[:, None] >= lts[None, :]
    elif mode == 4:
        lte = tl.load(
            lt_end_ptr + base_offset + k_offsets, mask=load_mask, other=pad_lt
        )
        uts = tl.load(
            ut_start_ptr + base_offset + k_offsets, mask=load_mask, other=pad_ut
        )
        ute = tl.load(
            ut_end_ptr + base_offset + k_offsets, mask=load_mask, other=pad_ut
        )
        dense_mask = (
            (q_rows[:, None] >= lts[None, :]) & (q_rows[:, None] < lte[None, :])
        ) | (
            (q_rows[:, None] >= uts[None, :]) & (q_rows[:, None] < ute[None, :])
        )
    else:
        if causal:
            lte = tl.load(
                lt_end_ptr + base_offset + k_offsets,
                mask=load_mask,
                other=pad_lt,
            )
            dense_mask = (q_rows[:, None] >= lts[None, :]) & (
                q_rows[:, None] < lte[None, :]
            )
        else:
            ute = tl.load(
                ut_end_ptr + base_offset + k_offsets,
                mask=load_mask,
                other=pad_ut,
            )
            dense_mask = (q_rows[:, None] >= lts[None, :]) | (
                q_rows[:, None] < ute[None, :]
            )

    X = (1.0 - dense_mask) * X  # set 0 for sum reduce
    return X, dense_mask


@triton.jit
def check_dense_contains_partial_stride(
    dense_flashmask,
    q_token_mask,
    k_token_mask,
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    dense_flashmask = tl.where(
        (q_token_mask[:, None] & k_token_mask[None, :]),
        dense_flashmask.to(tl.int32),
        tl.full([], 0, tl.int32),
    )
    mask_stride_cnt = dense_flashmask.reshape(
        BLOCK_SIZE // STRIDE, BLOCK_SIZE // STRIDE, STRIDE
    ).sum(2)
    mask_stride_valid_cnt = (
        k_token_mask.reshape(1, BLOCK_SIZE // STRIDE, STRIDE)
        .to(tl.int32)
        .sum(2)
    )

    mask_stride_is_partial = (mask_stride_cnt > 0) & (
        mask_stride_cnt < mask_stride_valid_cnt
    )
    # return mask_stride_is_partial
    return tl.sum(mask_stride_is_partial.to(tl.int32)) > 0


@triton.jit
def gemm_fuse_softmax_causal(
    q,
    k,
    out,
    out_boundary_mask,
    # --- Mask Pointers ---
    lt_start_ptr,
    lt_end_ptr,
    ut_start_ptr,
    ut_end_ptr,
    lt_start_nstridemax,
    lt_start_nstridemin,
    lt_end_nstridemax,
    lt_end_nstridemin,
    ut_start_nstridemax,
    ut_start_nstridemin,
    ut_end_nstridemax,
    ut_end_nstridemin,
    # --- Params ---
    scale: float,
    seqlen_q: int,
    seqlen_k: int,
    num_q_blocks: int,
    num_k_blocks: int,
    stride_mask_b,
    stride_mask_h,
    indices_mask_b,
    indices_mask_h,
    N_STRIDES,
    STRIDE: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mode: tl.constexpr,
):
    i_block = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)

    ratio: tl.constexpr = BLOCK_SIZE // STRIDE

    # ================= 1. Coordinates Setup =================
    q_stride_base = i_block * ratio
    offs_q_stride = q_stride_base + tl.arange(0, ratio)

    mask_ptr_base_bh_stride = i_b * stride_mask_b + i_h * stride_mask_h
    mask_ptr_base_bh_tokens = i_b * indices_mask_b + i_h * indices_mask_h

    # Load Q
    p_q = q + i_b * seqlen_q * H * K + (i_block * BLOCK_SIZE) * H * K + i_h * K
    p_q = (
        p_q
        + tl.arange(0, ratio)[:, None] * (H * K * STRIDE)
        + tl.arange(0, K)[None, :]
        + H * K * (i_h % STRIDE)
    )
    offs_tokens_q = (
        tl.arange(0, ratio) * STRIDE + i_block * BLOCK_SIZE + (i_h % STRIDE)
    )  # round-robin offset
    mask_q = offs_tokens_q < seqlen_q
    # mask_q = offs_tokens_q[:, None] < seqlen_q

    b_q = tl.load(p_q, mask=mask_q[:, None], other=0.0)
    b_q = (b_q * scale).to(b_q.dtype)

    # Softmax Accumulators
    m_i = tl.full([ratio], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([ratio], dtype=tl.float32)

    # Causal / FA3 Setup
    shift = seqlen_k - seqlen_q

    # k_safe_end: K blocks strictly to the left of the diagonal (Safe to compute fully)
    # Condition: k_block_end <= q_block_start + shift
    # (k + 1) * BLOCK <= i_block * BLOCK + shift
    k_safe_end = (i_block * BLOCK_SIZE + shift) // BLOCK_SIZE
    k_safe_end = min(num_k_blocks, max(0, k_safe_end))

    # k_valid_end: The last K block that intersects with the diagonal or Q block
    # Condition: k_block_start <= q_block_end_idx + shift
    # k * BLOCK <= ((i_block + 1) * BLOCK - 1) + shift
    k_valid_end = ((i_block + 1) * BLOCK_SIZE - 1 + shift) // BLOCK_SIZE + 1
    k_valid_end = min(num_k_blocks, max(k_safe_end, k_valid_end))

    p_k_base = k + i_b * seqlen_k * H * K + i_h * K
    offs_k_base = tl.arange(0, K)[:, None]
    offs_stride_k = tl.arange(0, ratio)
    offs_tokens_k = tl.arange(0, BLOCK_SIZE)

    # ================= 2. Loop 1: Statistics =================
    for iter in range(0, k_safe_end):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=True,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            # Load K & Compute Dot
            p_k = (
                p_k_base
                + iter * BLOCK_SIZE * H * K
                + tl.arange(0, BLOCK_SIZE)[None, :] * H * K
                + offs_k_base
            )
            b_k = tl.load(p_k)
            # CHANGE: NO REDUCE HERE
            # logits = tl.dot(b_q, b_k) # [ratio, BLOCK_SIZE]

            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=True,
                mode=mode,
            )

            real_partially_masked_stride_mask = (
                ~fully_masked_stride_mask
            ) & partially_masked_stride_mask
            if tl.sum(real_partially_masked_stride_mask) > 0:
                logits = tl.dot(b_q, b_k)  # [ratio, BLOCK_SIZE]
                curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE
                curr_token_load_mask = (
                    iter * BLOCK_SIZE + offs_tokens_k
                ) < seqlen_k
                X, dense_flashmask = flashmask_apply(
                    logits,
                    offs_tokens_q,
                    curr_token_offset,
                    offs_tokens_k,
                    curr_token_load_mask,
                    lt_start_ptr,
                    lt_end_ptr,
                    ut_start_ptr,
                    ut_end_ptr,
                    causal=True,
                    mode=mode,
                )

                # Reduce token logits to get stride score
                X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)
                fully_masked_by_fm = (
                    dense_flashmask.reshape(ratio, ratio, STRIDE)
                ).min(axis=2) == 1
                X = tl.where(fully_masked_by_fm, -1.0e6, X)
            else:
                # Reduce token logits to get stride score
                X = tl.dot(b_q, b_k.reshape(K, ratio, STRIDE).sum(2))

            # Explicitly mask out fully masked stride blocks
            X = tl.where(fully_masked_stride_mask, -1.0e6, X)
            # if i_block == 0 and iter == 0:
            #     tl.device_print("stride_logits", X / 1.4426950408889634)

            # Update Stats
            m_local = tl.max(X, 1)
            m_new = tl.maximum(m_i, m_local)
            alpha = tl.math.exp2(m_i - m_new)
            X = X - m_new[:, None]
            l_local = tl.sum(tl.math.exp2(X), 1)
            l_i = l_i * alpha + l_local
            m_i = m_new

    for iter in range(k_safe_end, k_valid_end):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES
        # k_col_min = iter * BLOCK_SIZE

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=True,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            p_k = (
                p_k_base
                + iter * BLOCK_SIZE * H * K
                + tl.arange(0, BLOCK_SIZE)[None, :] * H * K
                + offs_k_base
            )
            mask_k = (
                tl.arange(0, BLOCK_SIZE)[None, :] + iter * BLOCK_SIZE
            ) < seqlen_k
            b_k = tl.load(p_k, mask=mask_k, other=0.0)
            # b_k = b_k.reshape(K, ratio, STRIDE)
            # b_k = tl.sum(b_k, axis=2)
            logits = tl.dot(b_q, b_k)

            curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE
            curr_token_load_mask = (
                iter * BLOCK_SIZE + offs_tokens_k
            ) < seqlen_k
            X, dense_flashmask = flashmask_apply(
                logits,
                offs_tokens_q,
                curr_token_offset,
                offs_tokens_k,
                curr_token_load_mask,
                lt_start_ptr,
                lt_end_ptr,
                ut_start_ptr,
                ut_end_ptr,
                causal=True,
                mode=mode,
            )
            global_offs_k = iter * BLOCK_SIZE + offs_tokens_k

            # Causal Condition: k_idx > q_idx + shift => Masked
            # Mask value: 0.0 (Identity for sum reduction of logits)
            causal_mask_token = global_offs_k[None, :] > (
                offs_tokens_q[:, None] + shift
            )
            X = tl.where(causal_mask_token, 0.0, X)

            # Reduce token logits to get stride score
            X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)
            # fully_masked_by_fm = dense_flashmask.reshape(ratio, ratio, STRIDE).min(axis=2) == 1
            causal_fuse_fm = causal_mask_token | dense_flashmask
            fully_masked_by_fm = (
                causal_fuse_fm.reshape(ratio, ratio, STRIDE).min(axis=2) == 1
            )
            X = tl.where(fully_masked_by_fm, -1.0e6, X)

            offs_k_stride_global = iter * ratio + offs_stride_k
            k_stride_token_start = offs_k_stride_global * STRIDE
            visibility_limit = offs_tokens_q + shift
            causal_mask_stride = (
                k_stride_token_start[None, :] > visibility_limit[:, None]
            )

            X = tl.where(causal_mask_stride, -1.0e6, X)
            # Explicitly mask out fully masked stride blocks
            X = tl.where(fully_masked_stride_mask, -1.0e6, X)

            m_local = tl.max(X, 1)
            m_new = tl.maximum(m_i, m_local)
            alpha = tl.math.exp2(m_i - m_new)
            X = X - m_new[:, None]
            l_local = tl.sum(tl.math.exp2(X), 1)
            l_i = l_i * alpha + l_local
            m_i = m_new

    # ================= 3. Output Preparation =================
    l_i_inv = 1.0 / l_i

    stride_out_b = (H * num_q_blocks * num_k_blocks).to(tl.int64)
    stride_out_head = (num_q_blocks * num_k_blocks).to(tl.int64)
    stride_out_q = num_k_blocks.to(tl.int64)
    p_out = (
        out
        + i_b * stride_out_b
        + i_h * stride_out_head
        + i_block * stride_out_q
    )
    p_out_mask = (
        out_boundary_mask
        + i_b * stride_out_b
        + i_h * stride_out_head
        + i_block * stride_out_q
    )

    # ================= 4. Loop 2: Output (Exact Mirror) =================
    # 4.1 Non-Causal Blocks
    for iter in range(0, k_safe_end):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=True,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            # Load K & Compute Dot
            p_k = (
                p_k_base
                + iter * BLOCK_SIZE * H * K
                + tl.arange(0, BLOCK_SIZE)[None, :] * H * K
                + offs_k_base
            )
            b_k = tl.load(p_k)

            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=True,
                mode=mode,
            )

            real_partially_masked_stride_mask = (
                ~fully_masked_stride_mask
            ) & partially_masked_stride_mask

            if tl.sum(real_partially_masked_stride_mask) > 0:
                logits = tl.dot(b_q, b_k)  # [ratio, BLOCK_SIZE]

                curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE

                curr_token_load_mask = (
                    iter * BLOCK_SIZE + offs_tokens_k
                ) < seqlen_k

                X, dense_flashmask = flashmask_apply(
                    logits,
                    offs_tokens_q,
                    curr_token_offset,
                    offs_tokens_k,
                    curr_token_load_mask,
                    lt_start_ptr,
                    lt_end_ptr,
                    ut_start_ptr,
                    ut_end_ptr,
                    causal=True,
                    mode=mode,
                )

                # Reduce token logits to get stride score

                X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)

                fully_masked_by_fm = (
                    dense_flashmask.reshape(ratio, ratio, STRIDE).min(axis=2)
                    == 1
                )

                X = tl.where(fully_masked_by_fm, -1.0e6, X)

                # partial_stride_mask = check_dense_contains_partial_stride(dense_flashmask, curr_token_load_mask, BLOCK_SIZE, STRIDE):

                if check_dense_contains_partial_stride(
                    dense_flashmask,
                    q_token_mask=mask_q,  # [ratio]
                    k_token_mask=curr_token_load_mask,  # [block_size]
                    BLOCK_SIZE=BLOCK_SIZE,
                    STRIDE=STRIDE,
                ):
                    tl.store(p_out_mask + iter, tl.full([], 1, dtype=tl.int8))

            else:
                # Reduce token logits to get stride score

                X = tl.dot(b_q, b_k.reshape(K, ratio, STRIDE).sum(2))

            X = tl.where(fully_masked_stride_mask, -1.0e6, X)

            # Normalization & Reduction
            X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
            X = tl.where(mask_q[:, None], X, 0)
            X = tl.where(m_i[:, None] < -1.0e5, 0, X)
            X = tl.sum(X, 1)  # Sum K-strides
            X = tl.sum(X, 0)  # Sum Q-tokens
            tl.store(p_out + iter, X.to(out.type.element_ty))

    # 4.2 Causal Block

    for iter in range(k_safe_end, k_valid_end):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio

        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        # k_col_min = iter * BLOCK_SIZE

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=True,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            p_k = (
                p_k_base
                + iter * BLOCK_SIZE * H * K
                + tl.arange(0, BLOCK_SIZE)[None, :] * H * K
                + offs_k_base
            )

            mask_k = (
                tl.arange(0, BLOCK_SIZE)[None, :] + iter * BLOCK_SIZE
            ) < seqlen_k

            b_k = tl.load(p_k, mask=mask_k, other=0.0)

            logits = tl.dot(b_q, b_k)
            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=True,
                mode=mode,
            )

            real_partially_masked_stride_mask = (
                ~fully_masked_stride_mask
            ) & partially_masked_stride_mask

            curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE

            curr_token_load_mask = (
                iter * BLOCK_SIZE + offs_tokens_k
            ) < seqlen_k

            X, dense_flashmask = flashmask_apply(
                logits,
                offs_tokens_q,
                curr_token_offset,
                offs_tokens_k,
                curr_token_load_mask,
                lt_start_ptr,
                lt_end_ptr,
                ut_start_ptr,
                ut_end_ptr,
                causal=True,
                mode=mode,
            )

            global_offs_k = iter * BLOCK_SIZE + offs_tokens_k

            # Causal Condition: k_idx > q_idx + shift => Masked

            # Mask value: 0.0 (Identity for sum reduction of logits)

            causal_mask_token = global_offs_k[None, :] > (
                offs_tokens_q[:, None] + shift
            )

            X = tl.where(causal_mask_token, 0.0, X)

            # Reduce token logits to get stride score
            X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)

            causal_fuse_fm = causal_mask_token | dense_flashmask
            fully_masked_by_fm = (
                causal_fuse_fm.reshape(ratio, ratio, STRIDE).min(axis=2) == 1
            )

            X = tl.where(fully_masked_by_fm, -1.0e6, X)

            if check_dense_contains_partial_stride(
                causal_fuse_fm,
                q_token_mask=mask_q,
                k_token_mask=curr_token_load_mask,
                BLOCK_SIZE=BLOCK_SIZE,
                STRIDE=STRIDE,
            ):
                tl.store(p_out_mask + iter, tl.full([], 1, dtype=tl.int8))

            offs_k_stride_global = iter * ratio + offs_stride_k
            k_stride_token_start = offs_k_stride_global * STRIDE
            visibility_limit = offs_tokens_q + shift
            causal_mask_stride = (
                k_stride_token_start[None, :] > visibility_limit[:, None]
            )
            X = tl.where(causal_mask_stride, -1.0e6, X)

            # Explicitly mask out fully masked stride blocks
            X = tl.where(fully_masked_stride_mask, -1.0e6, X)
            X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
            X = tl.where(m_i[:, None] < -1.0e5, 0, X)
            X = tl.where(mask_q[:, None], X, 0)
            X = tl.sum(X, 1)
            X = tl.sum(X, 0)
            tl.store(p_out + iter, X.to(out.type.element_ty))


@triton.jit
def gemm_fuse_softmax_non_causal(
    q,
    k,
    out,
    out_boundary_mask,
    # --- Mask Pointers ---
    lt_start_ptr,
    lt_end_ptr,
    ut_start_ptr,
    ut_end_ptr,
    lt_start_nstridemax,
    lt_start_nstridemin,
    lt_end_nstridemax,
    lt_end_nstridemin,
    ut_start_nstridemax,
    ut_start_nstridemin,
    ut_end_nstridemax,
    ut_end_nstridemin,
    # --- Params ---
    scale: float,
    seqlen_q: int,
    seqlen_k: int,
    num_q_blocks: int,
    num_k_blocks: int,
    stride_mask_b,
    stride_mask_h,
    indices_mask_b,
    indices_mask_h,
    N_STRIDES,
    STRIDE: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    mode: tl.constexpr,
):
    """
    Non-Causal (Bidirectional) Version:
    1. Loop over ALL K blocks (0 to num_k_blocks).
    2. No "Diagonal/Causal" check logic.
    3. Block Mask logic remains active (controlled by causal=False).
    """

    i_block = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)
    ratio: tl.constexpr = BLOCK_SIZE // STRIDE

    # ================= 1. Coordinates Setup =================

    q_stride_base = i_block * ratio
    offs_q_stride = q_stride_base + tl.arange(0, ratio)
    mask_ptr_base_bh_stride = i_b * stride_mask_b + i_h * stride_mask_h
    mask_ptr_base_bh_tokens = i_b * indices_mask_b + i_h * indices_mask_h
    # Load Q (Round-Robin Sampling)

    p_q = q + i_b * seqlen_q * H * K + (i_block * BLOCK_SIZE) * H * K + i_h * K

    p_q = (
        p_q
        + tl.arange(0, ratio)[:, None] * (H * K * STRIDE)
        + tl.arange(0, K)[None, :]
        + H * K * (i_h % STRIDE)
    )

    offs_tokens_q = (
        tl.arange(0, ratio) * STRIDE + i_block * BLOCK_SIZE + (i_h % STRIDE)
    )

    mask_q = offs_tokens_q < seqlen_q
    b_q = tl.load(p_q, mask=mask_q[:, None], other=0.0)
    b_q = (b_q * scale).to(b_q.dtype)

    # Softmax Accumulators
    m_i = tl.full([ratio], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([ratio], dtype=tl.float32)

    # K Pointers Setup
    p_k_base = k + i_b * seqlen_k * H * K + i_h * K
    offs_k_base = tl.arange(0, K)[:, None]
    offs_stride_k = tl.arange(0, ratio)
    offs_tokens_k = tl.arange(0, BLOCK_SIZE)
    # ================= 2. Loop 1: Statistics =================
    # Iterate over ALL K blocks (No causal split)

    for iter in range(0, num_k_blocks):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        # [Check Fully Masked]
        # causal=False affects logic inside check (e.g. loads UT bounds)
        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=False,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            # Load K & Compute Dot
            p_k = (
                p_k_base
                + iter * BLOCK_SIZE * H * K
                + tl.arange(0, BLOCK_SIZE)[None, :] * H * K
                + offs_k_base
            )

            mask_k = (
                tl.arange(0, BLOCK_SIZE)[None, :] + iter * BLOCK_SIZE
            ) < seqlen_k

            b_k = tl.load(p_k, mask=mask_k, other=0.0)
            # Compute Scores: [ratio, K] @ [K, BLOCK_SIZE] -> [ratio, BLOCK_SIZE]
            # logits = tl.dot(b_q, b_k)
            # [Check Partial Mask]

            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=False,
                mode=mode,
            )

            real_partially_masked_stride_mask = (
                ~fully_masked_stride_mask
            ) & partially_masked_stride_mask

            if tl.sum(real_partially_masked_stride_mask) > 0:
                logits = tl.dot(b_q, b_k)
                curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE
                curr_token_load_mask = (
                    iter * BLOCK_SIZE + offs_tokens_k
                ) < seqlen_k

                X, dense_flashmask = flashmask_apply(
                    logits,
                    offs_tokens_q,
                    curr_token_offset,
                    offs_tokens_k,
                    curr_token_load_mask,
                    lt_start_ptr,
                    lt_end_ptr,
                    ut_start_ptr,
                    ut_end_ptr,
                    causal=False,
                    mode=mode,
                )

                # Reduce token logits to get stride score
                X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)

                fully_masked_by_fm = (
                    dense_flashmask.reshape(ratio, ratio, STRIDE).min(axis=2)
                    == 1
                )
                X = tl.where(fully_masked_by_fm, -1.0e6, X)

            else:
                X = tl.dot(b_q, b_k.reshape(K, ratio, STRIDE).sum(2))

            # Explicitly mask out fully masked stride blocks
            X = tl.where(fully_masked_stride_mask, -1.0e6, X)

            # Update Stats
            m_local = tl.max(X, 1)
            m_new = tl.maximum(m_i, m_local)
            alpha = tl.math.exp2(m_i - m_new)
            X = X - m_new[:, None]
            l_local = tl.sum(tl.math.exp2(X), 1)
            l_i = l_i * alpha + l_local
            m_i = m_new

    # ================= 3. Output Preparation =================

    l_i_inv = 1.0 / l_i
    stride_out_b = (H * num_q_blocks * num_k_blocks).to(tl.int64)
    stride_out_head = (num_q_blocks * num_k_blocks).to(tl.int64)
    stride_out_q = num_k_blocks.to(tl.int64)

    p_out = (
        out
        + i_b * stride_out_b
        + i_h * stride_out_head
        + i_block * stride_out_q
    )

    p_out_mask = (
        out_boundary_mask
        + i_b * stride_out_b
        + i_h * stride_out_head
        + i_block * stride_out_q
    )

    # ================= 4. Loop 2: Output (Exact Mirror) =================

    for iter in range(0, num_k_blocks):
        curr_stride_offset = mask_ptr_base_bh_stride + iter * ratio
        curr_load_mask = (iter * ratio + offs_stride_k) < N_STRIDES

        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_stride_k,
            curr_load_mask,
            offs_tokens_q,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=False,
            mode=mode,
        )

        if tl.sum(fully_masked_stride_mask.to(tl.int32)) < ratio * ratio:
            p_k = (
                p_k_base
                + iter * BLOCK_SIZE * H * K
                + tl.arange(0, BLOCK_SIZE)[None, :] * H * K
                + offs_k_base
            )
            mask_k = (
                tl.arange(0, BLOCK_SIZE)[None, :] + iter * BLOCK_SIZE
            ) < seqlen_k

            b_k = tl.load(p_k, mask=mask_k, other=0.0)
            # logits = tl.dot(b_q, b_k)

            partially_masked_stride_mask = check_partially_masked_state(
                curr_stride_offset,
                offs_stride_k,
                curr_load_mask,
                offs_tokens_q,
                lt_start_nstridemin,
                lt_end_nstridemax,
                ut_start_nstridemin,
                ut_end_nstridemax,
                causal=False,
                mode=mode,
            )

            real_partially_masked_stride_mask = (
                ~fully_masked_stride_mask
            ) & partially_masked_stride_mask

            if tl.sum(real_partially_masked_stride_mask) > 0:
                logits = tl.dot(b_q, b_k)
                curr_token_offset = mask_ptr_base_bh_tokens + iter * BLOCK_SIZE

                curr_token_load_mask = (
                    iter * BLOCK_SIZE + offs_tokens_k
                ) < seqlen_k

                X, dense_flashmask = flashmask_apply(
                    logits,
                    offs_tokens_q,
                    curr_token_offset,
                    offs_tokens_k,
                    curr_token_load_mask,
                    lt_start_ptr,
                    lt_end_ptr,
                    ut_start_ptr,
                    ut_end_ptr,
                    causal=False,
                    mode=mode,
                )
                # Reduce token logits to get stride score
                X = X.reshape(ratio, ratio, STRIDE).sum(axis=2)

                fully_masked_by_fm = (
                    dense_flashmask.reshape(ratio, ratio, STRIDE).min(axis=2)
                    == 1
                )
                X = tl.where(fully_masked_by_fm, -1.0e6, X)

                if check_dense_contains_partial_stride(
                    dense_flashmask,
                    q_token_mask=mask_q,
                    k_token_mask=curr_token_load_mask,
                    BLOCK_SIZE=BLOCK_SIZE,
                    STRIDE=STRIDE,
                ):
                    tl.store(p_out_mask + iter, tl.full([], 1, dtype=tl.int8))

            else:
                # Reduce token logits to get stride score
                X = tl.dot(b_q, b_k.reshape(K, ratio, STRIDE).sum(2))
            X = tl.where(fully_masked_stride_mask, -1.0e6, X)

            # Normalization & Reduction
            X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
            X = tl.where(mask_q[:, None], X, 0)
            X = tl.where(m_i[:, None] < -1.0e5, 0, X)
            X = tl.sum(X, 1)  # Sum K-strides
            X = tl.sum(X, 0)  # Sum Q-tokens
            tl.store(p_out + iter, X.to(out.type.element_ty))


def rr_attn_estimate_triton_func(
    q: paddle.Tensor,
    k: paddle.Tensor,
    startend_row_indices: paddle.Tensor,
    stride: int = 8,
    causal: bool = True,
) -> paddle.Tensor:
    bsz, q_len, num_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape
    num_indices_heads = startend_row_indices.shape[1]

    if num_heads != num_kv_heads:
        k = k.repeat_interleave(num_heads // num_kv_heads, axis=2).contiguous()

    if num_heads != num_indices_heads:
        startend_row_indices = startend_row_indices.repeat_interleave(
            num_heads // num_indices_heads, axis=1
        ).contiguous()

    # startend_row_indices shape assumption: [B, H, Q, Mode] or [B, 1, Q, Mode]
    # Ensure it's on the same device
    assert startend_row_indices.place == q.place
    # print(startend_row_indices)
    # mode = 1: [lt_start]
    # mode = 2: [, lt_end] (Causal)
    # mode = 4: [lt_start, lt_end, ut_start, ut_end]

    mode = startend_row_indices.shape[-1]
    assert mode in [1, 2, 4]
    chunk_size = stride
    # --- 1. Prepare Raw Pointers (Token Level) ---
    # We need these for flashmask_apply inside the kernel
    lt_start_raw = startend_row_indices[..., 0].contiguous()
    # Initialize optional raw pointers to lt_start_raw (as a safe dummy)
    # to avoid passing None/Invalid pointers to Kernel
    lt_end_raw = lt_start_raw
    ut_start_raw = lt_start_raw
    ut_end_raw = lt_start_raw
    if mode == 2:
        if causal:
            lt_end_raw = startend_row_indices[..., 1].contiguous()

        else:
            ut_end_raw = startend_row_indices[..., 1].contiguous()

    elif mode == 4:
        lt_end_raw = startend_row_indices[..., 1].contiguous()
        ut_start_raw = startend_row_indices[..., 2].contiguous()
        ut_end_raw = startend_row_indices[..., 3].contiguous()

    # --- 2. Calculate Strides for Raw Pointers ---
    # Used for indices_mask_b / indices_mask_h
    # Assuming startend_row_indices shape is [B, H, Q, Mode]
    indices_mask_b = lt_start_raw.strides[0]

    if lt_start_raw.shape[1] == 1 and num_heads > 1:
        # Broadcasting raw mask across heads
        indices_mask_h = 0

    else:
        indices_mask_h = lt_start_raw.strides[1]

    # --- 3. Prepare Min/Max Pointers (Stride/Block Level) ---
    # Helper to generate min/max views
    # (Assuming prepare_maxmin implementation exists and returns tensors on GPU)
    # Initialize all to None
    lt_start_nstridemax, lt_start_nstridemin = None, None
    lt_end_nstridemax, lt_end_nstridemin = None, None
    ut_start_nstridemax, ut_start_nstridemin = None, None
    ut_end_nstridemax, ut_end_nstridemin = None, None

    # LT Start (Always exists)
    lt_start_nstridemax, lt_start_nstridemin = prepare_maxmin(
        lt_start_raw, chunk_size
    )

    # Base tensor for safe_ptr (stride level)
    base_tensor_stride = lt_start_nstridemax
    if mode == 2:
        if causal:
            lt_end_nstridemax, lt_end_nstridemin = prepare_maxmin(
                lt_end_raw, chunk_size
            )
        else:
            ut_end_nstridemax, ut_end_nstridemin = prepare_maxmin(
                ut_end_raw, chunk_size
            )

    elif mode == 4:
        lt_end_nstridemax, lt_end_nstridemin = prepare_maxmin(
            lt_end_raw, chunk_size
        )

        ut_start_nstridemax, ut_start_nstridemin = prepare_maxmin(
            ut_start_raw, chunk_size
        )

        ut_end_nstridemax, ut_end_nstridemin = prepare_maxmin(
            ut_end_raw, chunk_size
        )

    def safe_ptr(t):
        return t if t is not None else base_tensor_stride

    # --- 4. Calculate Strides for Min/Max Pointers ---
    # Used for stride_mask_b / stride_mask_h
    stride_mask_b = base_tensor_stride.strides[0]

    if base_tensor_stride.shape[1] == 1 and num_heads > 1:
        stride_mask_h = 0

    else:
        stride_mask_h = base_tensor_stride.strides[1]

    n_strides_len = base_tensor_stride.shape[2]

    # --- 5. Kernel Launch Setup ---
    BLOCK_SIZE = 128
    num_q_blocks = triton.cdiv(q_len, BLOCK_SIZE)
    num_k_blocks = triton.cdiv(kv_len, BLOCK_SIZE)
    attn_sums = paddle.zeros(
        (bsz, num_heads, num_q_blocks, num_k_blocks),
        dtype=q.dtype,
    )

    boundary_protection_mask = paddle.zeros(
        (bsz, num_heads, num_q_blocks, num_k_blocks),
        dtype=paddle.bool,
    )

    grid = (num_q_blocks, num_heads, bsz)
    # print(grid)
    # print(lt_start_raw.shape)
    # print(lt_start_nstridemax.shape)

    scale = 1.4426950408889634 / math.sqrt(head_dim) / stride

    if causal:
        gemm_fuse_softmax_causal[grid](
            q,
            k,
            attn_sums,
            boundary_protection_mask,
            lt_start_raw,
            lt_end_raw,
            ut_start_raw,
            ut_end_raw,
            safe_ptr(lt_start_nstridemax),
            safe_ptr(lt_start_nstridemin),
            safe_ptr(lt_end_nstridemax),
            safe_ptr(lt_end_nstridemin),
            safe_ptr(ut_start_nstridemax),
            safe_ptr(ut_start_nstridemin),
            safe_ptr(ut_end_nstridemax),
            safe_ptr(ut_end_nstridemin),
            scale=scale,
            seqlen_q=q_len,
            seqlen_k=kv_len,
            num_q_blocks=num_q_blocks,
            num_k_blocks=num_k_blocks,
            stride_mask_b=stride_mask_b,
            stride_mask_h=stride_mask_h,
            indices_mask_b=indices_mask_b,
            indices_mask_h=indices_mask_h,
            N_STRIDES=n_strides_len,
            STRIDE=stride,
            H=num_heads,
            K=head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            mode=mode,
        )

    else:
        gemm_fuse_softmax_non_causal[grid](
            q,
            k,
            attn_sums,
            boundary_protection_mask,
            lt_start_raw,
            lt_end_raw,
            ut_start_raw,
            ut_end_raw,
            safe_ptr(lt_start_nstridemax),
            safe_ptr(lt_start_nstridemin),
            safe_ptr(lt_end_nstridemax),
            safe_ptr(lt_end_nstridemin),
            safe_ptr(ut_start_nstridemax),
            safe_ptr(ut_start_nstridemin),
            safe_ptr(ut_end_nstridemax),
            safe_ptr(ut_end_nstridemin),
            scale=scale,
            seqlen_q=q_len,
            seqlen_k=kv_len,
            num_q_blocks=num_q_blocks,
            num_k_blocks=num_k_blocks,
            stride_mask_b=stride_mask_b,
            stride_mask_h=stride_mask_h,
            indices_mask_b=indices_mask_b,
            indices_mask_h=indices_mask_h,
            N_STRIDES=n_strides_len,
            STRIDE=stride,
            H=num_heads,
            K=head_dim,
            BLOCK_SIZE=BLOCK_SIZE,
            mode=mode,
        )

    return attn_sums, boundary_protection_mask
