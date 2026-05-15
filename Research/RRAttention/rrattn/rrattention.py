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

from __future__ import annotations

import math
from dataclasses import dataclass

import paddle
import paddle.nn.functional as F
import triton
import triton.language as tl
from paddle import use_compat_guard


@dataclass(frozen=True)
class RRAttnConfig:
    # GEMM q-stride tile. Larger tiles improve reuse but increase register pressure.
    block_m: int = 128
    # K-stride tile. Larger tiles improve K throughput but can reduce occupancy.
    block_n: int = 32
    # Triton launch parameters for the qchunk GEMM kernel.
    num_warps: int = 4
    num_stages: int = 1
    # K-stride segment size used by the softmax/reduce kernel.
    segment_size: int = 128


TUNABLE_FIELDS = (
    "block_m",
    "block_n",
    "num_warps",
    "num_stages",
    "segment_size",
)


def gpu_info() -> tuple[str, int | None]:
    if not paddle.device.is_compiled_with_cuda():
        return "", None
    try:
        device_name = paddle.device.cuda.get_device_name().lower()
        major, _ = paddle.device.cuda.get_device_capability()
        return device_name, major
    except Exception:
        return "", None


GPU_NAME, GPU_MAJOR = gpu_info()


def get_rrattn_config(
    head_dim: int, gpu_name: str | None = None
) -> RRAttnConfig:
    """Return the default estimate-kernel config.

    RRAttention currently fixes the public block size at 128, so block size is
    not a tuning dimension here. Tune the fields listed in TUNABLE_FIELDS per
    head_dim bucket.
    """
    gpu_name = (gpu_name or GPU_NAME or "").lower()

    if "h100" in gpu_name or "h800" in gpu_name:
        if head_dim <= 64:
            return RRAttnConfig(
                block_m=128,
                block_n=64,
                num_warps=4,
                num_stages=1,
                segment_size=256,
            )
        elif head_dim <= 128:
            return RRAttnConfig(
                block_m=128,
                block_n=64,
                num_warps=4,
                num_stages=3,
                segment_size=256,
            )
        else:
            return RRAttnConfig(
                block_m=128,
                block_n=64,
                num_warps=8,
                num_stages=1,
                segment_size=256,
            )

    if head_dim <= 64:
        return RRAttnConfig(
            block_m=128, block_n=16, num_warps=4, num_stages=1, segment_size=256
        )
    elif head_dim <= 128:
        return RRAttnConfig(
            block_m=128, block_n=32, num_warps=4, num_stages=2, segment_size=256
        )
    else:
        return RRAttnConfig(num_warps=4, num_stages=1)


LOG2E = 1.4426950408889634  # 1 / ln(2)
BLOCK_SIZE = 128


@dataclass
class RawPtrs:
    lt_start: paddle.Tensor
    lt_end: paddle.Tensor
    ut_start: paddle.Tensor
    ut_end: paddle.Tensor


@dataclass
class StrideMaxMinPtrs:
    lt_start_max: paddle.Tensor
    lt_start_min: paddle.Tensor
    lt_end_max: paddle.Tensor
    lt_end_min: paddle.Tensor
    ut_start_max: paddle.Tensor
    ut_start_min: paddle.Tensor
    ut_end_max: paddle.Tensor
    ut_end_min: paddle.Tensor
    n_strides: int


@dataclass
class MaskContext:
    mode: int
    stride_mm: StrideMaxMinPtrs
    num_indices_heads: int


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _normalize_config(
    config: RRAttnConfig | None, *, head_dim: int
) -> RRAttnConfig:
    if config is not None:
        return config
    return get_rrattn_config(head_dim)


@triton.jit
def scan_maxmin_chunked(
    input_ptr,
    output_max_ptr,
    output_min_ptr,
    seqlen,
    num_chunks,
    chunk_size: tl.constexpr,
    BN: tl.constexpr,
):
    INT_MAX: tl.constexpr = 2147483647
    INT_MIN: tl.constexpr = -2147483648

    i_tile = tl.program_id(0)
    i_bh = tl.program_id(1)

    p_tile = i_tile * BN + tl.arange(0, BN)
    mask_tile = p_tile < seqlen
    b_tile = tl.load(input_ptr + i_bh * seqlen + p_tile, mask=mask_tile)

    b_omax = tl.where(mask_tile, b_tile, INT_MIN).reshape(
        (BN // chunk_size, chunk_size)
    )
    b_omax = tl.max(b_omax, axis=1)

    b_omin = tl.where(mask_tile, b_tile, INT_MAX).reshape(
        (BN // chunk_size, chunk_size)
    )
    b_omin = tl.min(b_omin, axis=1)

    offs_out = tl.arange(0, BN // chunk_size) + i_tile * (BN // chunk_size)
    mask_out = offs_out < num_chunks
    tl.store(
        output_max_ptr + i_bh * num_chunks + offs_out, b_omax, mask=mask_out
    )
    tl.store(
        output_min_ptr + i_bh * num_chunks + offs_out, b_omin, mask=mask_out
    )


def prepare_maxmin(
    input_tensor: paddle.Tensor,
    chunk_size: int,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    bsz, num_heads, seq_len = input_tensor.shape
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    output_max = paddle.empty(
        [bsz, num_heads, num_chunks],
        dtype=paddle.int32,
        device=input_tensor.device,
    )
    output_min = paddle.empty(
        [bsz, num_heads, num_chunks],
        dtype=paddle.int32,
        device=input_tensor.device,
    )

    bn = 512
    grid = ((seq_len + bn - 1) // bn, bsz * num_heads)
    scan_maxmin_chunked[grid](
        input_tensor,
        output_max,
        output_min,
        seq_len,
        num_chunks,
        chunk_size=chunk_size,
        BN=bn,
    )
    return output_max, output_min


@triton.jit
def _load_bounds(
    base_offset,
    k_offsets,
    load_mask,
    ptr_start_lt,
    ptr_end_lt,
    ptr_start_ut,
    ptr_end_ut,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    INT_MAX: tl.constexpr = 2147483647
    INT_MIN: tl.constexpr = -2147483648

    pad_lt = INT_MAX
    pad_ut = INT_MIN

    b_lts = tl.load(
        ptr_start_lt + base_offset + k_offsets, mask=load_mask, other=pad_lt
    )

    need_lte: tl.constexpr = (causal and mode == 2) or (
        not causal and mode == 4
    )
    if need_lte:
        b_lte = tl.load(
            ptr_end_lt + base_offset + k_offsets, mask=load_mask, other=pad_lt
        )
    else:
        b_lte = tl.full(b_lts.shape, pad_lt, dtype=tl.int32)

    if causal:
        b_uts = tl.full(b_lts.shape, pad_ut, dtype=tl.int32)
    else:
        if mode == 4:
            b_uts = tl.load(
                ptr_start_ut + base_offset + k_offsets,
                mask=load_mask,
                other=pad_ut,
            )
        else:
            b_uts = tl.full(b_lts.shape, pad_ut, dtype=tl.int32)

    need_ute: tl.constexpr = (not causal) and (mode == 2 or mode == 4)
    if need_ute:
        b_ute = tl.load(
            ptr_end_ut + base_offset + k_offsets, mask=load_mask, other=pad_ut
        )
    else:
        b_ute = tl.full(b_lts.shape, pad_ut, dtype=tl.int32)

    return b_lts, b_lte, b_uts, b_ute


@triton.jit
def _is_block_fully_masked(block_rows, lts_max, lte_min, uts_max, ute_min):
    in_lt = (block_rows[:, None] >= lts_max[None, :]) & (
        block_rows[:, None] < lte_min[None, :]
    )
    in_ut = (block_rows[:, None] >= uts_max[None, :]) & (
        block_rows[:, None] < ute_min[None, :]
    )
    return in_lt | in_ut


@triton.jit
def check_fully_masked_state(
    mask_ptr_base_offset,
    k_offsets,
    k_load_mask,
    q_rows,
    ptrs_strict_lt_start,
    ptrs_strict_lt_end,
    ptrs_strict_ut_start,
    ptrs_strict_ut_end,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    fm_lts, fm_lte, fm_uts, fm_ute = _load_bounds(
        mask_ptr_base_offset,
        k_offsets,
        k_load_mask,
        ptrs_strict_lt_start,
        ptrs_strict_lt_end,
        ptrs_strict_ut_start,
        ptrs_strict_ut_end,
        causal=causal,
        mode=mode,
    )
    fm_geo = _is_block_fully_masked(q_rows, fm_lts, fm_lte, fm_uts, fm_ute)
    return fm_geo | (~k_load_mask[None, :])


@triton.jit
def _is_block_partially_masked(block_rows, lts_min, lte_max, uts_min, ute_max):
    overlap_lt = (block_rows[:, None] < lte_max[None, :]) & (
        block_rows[:, None] >= lts_min[None, :]
    )
    overlap_ut = (block_rows[:, None] < ute_max[None, :]) & (
        block_rows[:, None] >= uts_min[None, :]
    )
    return overlap_lt | overlap_ut


@triton.jit
def check_partially_masked_state(
    mask_ptr_base_offset,
    k_offsets,
    k_load_mask,
    q_rows,
    ptrs_perm_lt_start,
    ptrs_perm_lt_end,
    ptrs_perm_ut_start,
    ptrs_perm_ut_end,
    causal: tl.constexpr,
    mode: tl.constexpr,
):
    pm_lts, pm_lte, pm_uts, pm_ute = _load_bounds(
        mask_ptr_base_offset,
        k_offsets,
        k_load_mask,
        ptrs_perm_lt_start,
        ptrs_perm_lt_end,
        ptrs_perm_ut_start,
        ptrs_perm_ut_end,
        causal=causal,
        mode=mode,
    )
    return _is_block_partially_masked(q_rows, pm_lts, pm_lte, pm_uts, pm_ute)


@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)

    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(
        tl.sum(tl.where(mask == 0, y, 0), 1)[:, None, :], shape
    ).to(y.dtype)
    right = tl.broadcast_to(
        tl.sum(tl.where(mask == 1, y, 0), 1)[:, None, :], shape
    ).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)

    idtype = tl.core.get_int_dtype(
        bitwidth=x.dtype.primitive_bitwidth, signed=True
    )
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)

    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order

    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def bitonic_argsort_device(
    x, ids, n_dims: tl.constexpr, descending: tl.constexpr = tl.core.CONSTEXPR_0
):
    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(
            x, ids, i, 2 if i < n_dims else descending, n_dims
        )
    return x, ids


@triton.jit
def top_p_kernel(
    X_ptr,
    Out_ptr,
    stride_row,
    threshold_p,
    N_COLS,
    BLOCK_SIZE: tl.constexpr,
    NUM_DIMS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start_ptr = X_ptr + pid * stride_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask_load = offsets < N_COLS

    x_raw = tl.load(row_start_ptr + offsets, mask=mask_load, other=0.0).to(
        tl.float32
    )
    row_sum = tl.sum(x_raw, axis=0)

    out_row_ptr = Out_ptr + pid * stride_row
    if row_sum == 0.0:
        tl.store(
            out_row_ptr + offsets,
            tl.zeros([BLOCK_SIZE], dtype=tl.int8),
            mask=mask_load,
        )
        return

    actual_cutoff = row_sum * threshold_p
    padding_val = float("-inf")
    x_for_sort = tl.where(mask_load, x_raw, padding_val)
    ids = tl.arange(0, BLOCK_SIZE)

    x_sorted, ids_sorted = bitonic_argsort_device(
        x_for_sort, ids, NUM_DIMS, descending=1
    )

    cum_probs = tl.cumsum(x_sorted, axis=0)
    mask_keep = (cum_probs - x_sorted) < actual_cutoff
    mask_keep = mask_keep & (x_sorted > padding_val)

    mask_store = ids_sorted < N_COLS
    tl.store(out_row_ptr + ids_sorted, mask_keep.to(tl.int8), mask=mask_store)


def find_blocks_topp(x: paddle.Tensor, p: float) -> paddle.Tensor:
    original_shape = x.shape
    n = original_shape[-1]
    x_reshaped = x.reshape([-1, n]).contiguous()
    rows = x_reshaped.shape[0]

    block_size = triton.next_power_of_2(n)
    if block_size < 1:
        block_size = 1
    num_dims = int(math.log2(block_size))

    output_mask = paddle.empty(
        x_reshaped.shape, dtype=paddle.int8, device=x.device
    )
    top_p_kernel[(rows,)](
        x_reshaped,
        output_mask,
        x_reshaped.stride(0),
        p,
        n,
        BLOCK_SIZE=block_size,
        NUM_DIMS=num_dims,
    )
    return output_mask.astype(paddle.bool).reshape(original_shape)


def _build_fa3_causal_block_visible_mask(
    input_tensor: paddle.Tensor,
    q_len: int,
    kv_len: int,
) -> paddle.Tensor:
    batch_size, head_num, chunk_num, block_num = input_tensor.shape
    q_ids = paddle.arange(
        chunk_num, dtype=paddle.int32, device=input_tensor.device
    ).reshape([1, 1, chunk_num, 1])
    k_ids = paddle.arange(
        block_num, dtype=paddle.int32, device=input_tensor.device
    ).reshape([1, 1, 1, block_num])
    q_block_end = (
        paddle.minimum(
            (q_ids + 1) * BLOCK_SIZE,
            paddle.full(
                [1, 1, chunk_num, 1],
                q_len,
                dtype=paddle.int32,
                device=input_tensor.device,
            ),
        )
        - 1
    )
    k_block_start = k_ids * BLOCK_SIZE
    return k_block_start <= q_block_end + (kv_len - q_len)


def _build_causal_prefill_mandatory_mask(
    input_tensor: paddle.Tensor,
    q_len: int,
    kv_len: int,
) -> paddle.Tensor:
    batch_size, head_num, chunk_num, block_num = input_tensor.shape
    mask = paddle.zeros(
        [batch_size, head_num, chunk_num, block_num],
        dtype=paddle.bool,
        device=input_tensor.device,
    )
    mask[:, :, :, 0] = True

    q_ids = paddle.arange(
        chunk_num, dtype=paddle.int32, device=input_tensor.device
    )
    q_block_end = (
        paddle.minimum(
            (q_ids + 1) * BLOCK_SIZE,
            paddle.full(
                [chunk_num],
                q_len,
                dtype=paddle.int32,
                device=input_tensor.device,
            ),
        )
        - 1
    )
    diag_k_ids = (q_block_end + (kv_len - q_len)) // BLOCK_SIZE
    q_valid = q_block_end >= 0
    diag_valid = q_valid & (diag_k_ids >= 0) & (diag_k_ids < block_num)
    if bool(paddle.any(diag_valid).item()):
        q_indices = paddle.nonzero(diag_valid).flatten()
        k_indices = paddle.gather(diag_k_ids, q_indices).astype(paddle.int64)
        diag_mask = paddle.zeros_like(mask)
        diag_mask[:, :, q_indices, k_indices] = True
        mask = paddle.logical_or(mask, diag_mask)
    return mask


@triton.jit
def rrattn_gemm_qchunk_gqa_kernel(
    Q,
    K,
    Out,
    seqlen_q,
    seqlen_k,
    chunk_q_start,
    chunk_q_strides,
    n_k_strides,
    shift_tokens,
    out_stride_b,
    out_stride_h,
    out_stride_q,
    HQ: tl.constexpr,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GQA_HEADS_PER_CTA: tl.constexpr,
    is_causal: tl.constexpr,
):
    block_m = tl.program_id(0).to(tl.int64)
    block_n = tl.program_id(1).to(tl.int64)
    i_bhg = tl.program_id(2).to(tl.int64)

    G: tl.constexpr = HQ // H
    GROUPS_PER_KV: tl.constexpr = (
        G + GQA_HEADS_PER_CTA - 1
    ) // GQA_HEADS_PER_CTA

    i_b = i_bhg // (H * GROUPS_PER_KV)
    rem = i_bhg % (H * GROUPS_PER_KV)
    i_hkv = rem // GROUPS_PER_KV
    i_group = rem % GROUPS_PER_KV
    q_head_base = i_hkv * G + i_group * GQA_HEADS_PER_CTA

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    global_q_stride_start = chunk_q_start + block_m * BLOCK_M
    q_strides_global = global_q_stride_start + offs_m
    q_rows_local = block_m * BLOCK_M + offs_m
    k_strides_global = block_n * BLOCK_N + offs_n

    if is_causal:
        q_stride_end = chunk_q_start + (block_m + 1) * BLOCK_M
        max_q_token = (q_stride_end - 1) * STRIDE + (STRIDE - 1) + shift_tokens
        if max_q_token < block_n * BLOCK_N * STRIDE:
            return

    k_token_base = k_strides_global * STRIDE
    b_k = tl.zeros([HEAD_DIM, BLOCK_N], dtype=tl.float32)
    for lane in tl.static_range(STRIDE):
        k_token_ids = k_token_base + lane
        k_valid = k_token_ids < seqlen_k
        p_k = (
            K
            + i_b * (seqlen_k * H * HEAD_DIM).to(tl.int64)
            + k_token_ids[None, :] * (H * HEAD_DIM)
            + i_hkv * HEAD_DIM
            + tl.arange(0, HEAD_DIM)[:, None]
        )
        b_k += tl.load(p_k, mask=k_valid[None, :], other=0.0).to(tl.float32)

    k_oob = k_strides_global >= n_k_strides
    q_oob = q_rows_local >= chunk_q_strides

    for g_local in tl.static_range(GQA_HEADS_PER_CTA):
        q_head = q_head_base + g_local
        q_head_valid = q_head < ((i_hkv + 1) * G)
        q_head_safe = tl.minimum(q_head, HQ - 1)
        head_offset = q_head % STRIDE
        q_token_ids = q_strides_global * STRIDE + head_offset
        q_valid = (q_token_ids < seqlen_q) & q_head_valid

        p_q = (
            Q
            + i_b * (seqlen_q * HQ * HEAD_DIM).to(tl.int64)
            + q_token_ids[:, None] * (HQ * HEAD_DIM)
            + q_head_safe * HEAD_DIM
            + tl.arange(0, HEAD_DIM)[None, :]
        )
        b_q = tl.load(p_q, mask=q_valid[:, None], other=0.0)
        o = tl.dot(b_q, b_k.to(b_q.dtype))
        o = tl.where(k_oob[None, :], -1.0e6, o)
        o = tl.where(q_oob[:, None], -1.0e6, o)

        p_out = (
            Out
            + i_b * out_stride_b
            + q_head_safe * out_stride_h
            + q_rows_local[:, None] * out_stride_q
            + (block_n * BLOCK_N + offs_n)[None, :]
        )
        store_mask = (
            q_head_valid
            & (q_rows_local[:, None] < chunk_q_strides)
            & (k_strides_global[None, :] < n_k_strides)
        )
        tl.store(p_out, o.to(Out.type.element_ty), mask=store_mask)


@triton.jit
def rrattn_nomask_softmax_reduce_kernel(
    In,
    Out,
    OutBoundaryMask,
    scale,
    in_stride_b,
    in_stride_h,
    in_stride_q,
    out_stride_b,
    out_stride_h,
    out_stride_qb,
    chunk_q_start,
    chunk_q_strides,
    n_k_strides,
    num_q_blocks,
    num_k_blocks,
    seqlen_q,
    shift_tokens,
    STRIDE: tl.constexpr,
    ratio: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    i_qblock_local = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)

    q_stride_base_local = i_qblock_local * ratio
    q_stride_base_global = chunk_q_start + q_stride_base_local
    q_block_id = q_stride_base_global // ratio
    q_block_valid = (q_stride_base_local < chunk_q_strides) & (
        q_block_id < num_q_blocks
    )

    offs_q = tl.arange(0, ratio)
    q_rows_local = q_stride_base_local + offs_q
    q_valid = q_rows_local < chunk_q_strides

    head_offset = i_h % STRIDE
    q_token_ids = q_stride_base_global * STRIDE + offs_q * STRIDE + head_offset
    q_token_valid = q_token_ids < seqlen_q

    p_in_base = (
        In
        + i_b * in_stride_b
        + i_h * in_stride_h
        + q_stride_base_local * in_stride_q
    )
    p_out = (
        Out
        + i_b * out_stride_b
        + i_h * out_stride_h
        + q_block_id * out_stride_qb
    )
    p_out_mask = (
        OutBoundaryMask
        + i_b * out_stride_b
        + i_h * out_stride_h
        + q_block_id * out_stride_qb
    )

    m_i = tl.full([ratio], float("-inf"), dtype=tl.float32)
    l_i = tl.full([ratio], 1.0, dtype=tl.float32)

    num_segments = (n_k_strides + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    num_active_segments = num_segments
    if is_causal:
        last_q_token = (
            (q_stride_base_global + ratio - 1) * STRIDE
            + head_offset
            + shift_tokens
        )
        last_q_stride = last_q_token // STRIDE
        active_k_strides = tl.minimum(
            n_k_strides,
            tl.where(last_q_token >= 0, last_q_stride + 1, 0),
        )
        num_active_segments = (
            active_k_strides + SEGMENT_SIZE - 1
        ) // SEGMENT_SIZE
        diag_segment_idx = tl.maximum(last_q_stride, 0) // SEGMENT_SIZE

    for seg_idx in range(0, num_active_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        offs_k = tl.arange(0, SEGMENT_SIZE)
        p_in = (
            p_in_base
            + offs_q[:, None] * in_stride_q
            + (seg_start + offs_k)[None, :]
        )
        load_mask = q_valid[:, None] & (
            (seg_start + offs_k)[None, :] < n_k_strides
        )
        X = tl.load(p_in, mask=load_mask, other=-1.0e6).to(tl.float32) * scale
        if is_causal and seg_idx == diag_segment_idx:
            k_token_base = (seg_start + offs_k) * STRIDE
            causal_mask = (
                k_token_base[None, :] <= (q_token_ids + shift_tokens)[:, None]
            )
            X = tl.where(causal_mask, X, -1.0e6)

        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)
        l_local = tl.sum(tl.math.exp2(X - m_new[:, None]), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    l_i_inv = 1.0 / l_i

    BLOCKS_PER_SEG: tl.constexpr = SEGMENT_SIZE // ratio
    offs_kb = tl.arange(0, BLOCKS_PER_SEG)
    zero_mask = tl.zeros([BLOCKS_PER_SEG], dtype=tl.int8)

    for seg_idx in range(0, num_active_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        offs_k = tl.arange(0, SEGMENT_SIZE)
        p_in = (
            p_in_base
            + offs_q[:, None] * in_stride_q
            + (seg_start + offs_k)[None, :]
        )
        load_mask = q_valid[:, None] & (
            (seg_start + offs_k)[None, :] < n_k_strides
        )
        X = tl.load(p_in, mask=load_mask, other=-1.0e6).to(tl.float32) * scale
        if is_causal and seg_idx == diag_segment_idx:
            k_token_base = (seg_start + offs_k) * STRIDE
            causal_mask = (
                k_token_base[None, :] <= (q_token_ids + shift_tokens)[:, None]
            )
            X = tl.where(causal_mask, X, -1.0e6)
        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(m_i[:, None] < -1.0e5, 0.0, X)
        X = tl.where(q_token_valid[:, None], X, 0.0)

        X_reshaped = X.reshape(ratio, BLOCKS_PER_SEG, ratio)
        block_sums = tl.sum(tl.sum(X_reshaped, 2), 0)

        k_block_base = seg_start // ratio
        k_block_ids = k_block_base + offs_kb
        valid_store = q_block_valid & (k_block_ids < num_k_blocks)
        tl.store(
            p_out + k_block_ids,
            block_sums.to(Out.type.element_ty),
            mask=valid_store,
        )
        tl.store(
            p_out_mask + k_block_ids,
            zero_mask,
            mask=valid_store,
        )

    if is_causal:
        zero_vals = tl.zeros([BLOCKS_PER_SEG], dtype=tl.float32)
        for seg_idx in range(num_active_segments, num_segments):
            seg_start = seg_idx * SEGMENT_SIZE
            k_block_base = seg_start // ratio
            k_block_ids = k_block_base + offs_kb
            valid_store = q_block_valid & (k_block_ids < num_k_blocks)
            tl.store(
                p_out + k_block_ids,
                zero_vals.to(Out.type.element_ty),
                mask=valid_store,
            )
            tl.store(
                p_out_mask + k_block_ids,
                zero_mask,
                mask=valid_store,
            )


@triton.jit
def rrattn_flashmask_softmax_reduce_qchunk_kernel(
    In,
    Out,
    OutBoundaryMask,
    lt_start_nstridemax,
    lt_start_nstridemin,
    lt_end_nstridemax,
    lt_end_nstridemin,
    ut_start_nstridemax,
    ut_start_nstridemin,
    ut_end_nstridemax,
    ut_end_nstridemin,
    scale,
    in_stride_b,
    in_stride_h,
    in_stride_q,
    out_stride_b,
    out_stride_h,
    out_stride_qb,
    chunk_q_start,
    chunk_q_strides,
    n_k_strides,
    num_q_blocks,
    num_k_blocks,
    seqlen_q,
    shift_tokens,
    HQ: tl.constexpr,
    HIDS: tl.constexpr,
    STRIDE: tl.constexpr,
    ratio: tl.constexpr,
    SEGMENT_SIZE: tl.constexpr,
    mode: tl.constexpr,
    is_causal: tl.constexpr,
):
    i_qblock_local = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)

    GIDS: tl.constexpr = HQ // HIDS
    i_hid = i_h // GIDS

    q_stride_base_local = i_qblock_local * ratio
    q_stride_base_global = chunk_q_start + q_stride_base_local
    q_block_id = q_stride_base_global // ratio
    q_block_valid = (q_stride_base_local < chunk_q_strides) & (
        q_block_id < num_q_blocks
    )

    offs_q = tl.arange(0, ratio)
    q_rows_local = q_stride_base_local + offs_q
    q_valid = q_rows_local < chunk_q_strides
    q_strides_global = q_stride_base_global + offs_q

    head_offset = i_h % STRIDE
    q_token_ids = q_strides_global * STRIDE + head_offset
    q_token_valid = q_token_ids < seqlen_q

    p_in_base = (
        In
        + i_b * in_stride_b
        + i_h * in_stride_h
        + q_stride_base_local * in_stride_q
    )
    p_out = (
        Out
        + i_b * out_stride_b
        + i_h * out_stride_h
        + q_block_id * out_stride_qb
    )
    p_out_mask = (
        OutBoundaryMask
        + i_b * out_stride_b
        + i_h * out_stride_h
        + q_block_id * out_stride_qb
    )

    m_i = tl.full([ratio], float("-inf"), dtype=tl.float32)
    l_i = tl.full([ratio], 1.0, dtype=tl.float32)

    num_segments = (n_k_strides + SEGMENT_SIZE - 1) // SEGMENT_SIZE
    num_active_segments = num_segments
    if is_causal:
        last_q_token = (
            (q_stride_base_global + ratio - 1) * STRIDE
            + head_offset
            + shift_tokens
        )
        last_q_stride = last_q_token // STRIDE
        active_k_strides = tl.minimum(
            n_k_strides,
            tl.where(last_q_token >= 0, last_q_stride + 1, 0),
        )
        num_active_segments = (
            active_k_strides + SEGMENT_SIZE - 1
        ) // SEGMENT_SIZE

    for seg_idx in range(0, num_active_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        offs_k = tl.arange(0, SEGMENT_SIZE)
        p_in = (
            p_in_base
            + offs_q[:, None] * in_stride_q
            + (seg_start + offs_k)[None, :]
        )
        load_mask = q_valid[:, None] & (
            (seg_start + offs_k)[None, :] < n_k_strides
        )
        X = tl.load(p_in, mask=load_mask, other=-1.0e6).to(tl.float32) * scale
        if is_causal:
            k_token_base = (seg_start + offs_k) * STRIDE
            causal_mask = (
                k_token_base[None, :] <= (q_token_ids + shift_tokens)[:, None]
            )
            X = tl.where(causal_mask, X, -1.0e6)

        curr_stride_offset = (
            i_b * n_k_strides * HIDS + i_hid * n_k_strides + seg_start
        )
        curr_load_mask = (seg_start + offs_k) < n_k_strides
        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_k,
            curr_load_mask,
            q_token_ids,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=is_causal,
            mode=mode,
        )
        X = tl.where(fully_masked_stride_mask, -1.0e6, X)

        m_local = tl.max(X, 1)
        m_new = tl.maximum(m_i, m_local)
        alpha = tl.math.exp2(m_i - m_new)
        l_local = tl.sum(tl.math.exp2(X - m_new[:, None]), 1)
        l_i = l_i * alpha + l_local
        m_i = m_new

    l_i_inv = 1.0 / l_i

    BLOCKS_PER_SEG: tl.constexpr = SEGMENT_SIZE // ratio
    offs_kb = tl.arange(0, BLOCKS_PER_SEG)
    zero_mask = tl.zeros([BLOCKS_PER_SEG], dtype=tl.int8)

    for seg_idx in range(0, num_active_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        offs_k = tl.arange(0, SEGMENT_SIZE)
        p_in = (
            p_in_base
            + offs_q[:, None] * in_stride_q
            + (seg_start + offs_k)[None, :]
        )
        load_mask = q_valid[:, None] & (
            (seg_start + offs_k)[None, :] < n_k_strides
        )
        X = tl.load(p_in, mask=load_mask, other=-1.0e6).to(tl.float32) * scale

        causal_visible = tl.full([ratio, SEGMENT_SIZE], True, dtype=tl.int1)
        if is_causal:
            k_token_base = (seg_start + offs_k) * STRIDE
            causal_visible = (
                k_token_base[None, :] <= (q_token_ids + shift_tokens)[:, None]
            )
            X = tl.where(causal_visible, X, -1.0e6)

        curr_stride_offset = (
            i_b * n_k_strides * HIDS + i_hid * n_k_strides + seg_start
        )
        curr_load_mask = (seg_start + offs_k) < n_k_strides
        fully_masked_stride_mask = check_fully_masked_state(
            curr_stride_offset,
            offs_k,
            curr_load_mask,
            q_token_ids,
            lt_start_nstridemax,
            lt_end_nstridemin,
            ut_start_nstridemax,
            ut_end_nstridemin,
            causal=is_causal,
            mode=mode,
        )
        X = tl.where(fully_masked_stride_mask, -1.0e6, X)

        X = tl.exp2(X - m_i[:, None]) * l_i_inv[:, None]
        X = tl.where(m_i[:, None] < -1.0e5, 0.0, X)
        X = tl.where(q_token_valid[:, None], X, 0.0)

        X_reshaped = X.reshape(ratio, BLOCKS_PER_SEG, ratio)
        block_sums = tl.sum(tl.sum(X_reshaped, 2), 0)

        partially_masked_stride_mask = check_partially_masked_state(
            curr_stride_offset,
            offs_k,
            curr_load_mask,
            q_token_ids,
            lt_start_nstridemin,
            lt_end_nstridemax,
            ut_start_nstridemin,
            ut_end_nstridemax,
            causal=is_causal,
            mode=mode,
        )
        real_partial = (
            ~fully_masked_stride_mask
        ) & partially_masked_stride_mask
        if is_causal:
            real_partial = real_partial & causal_visible
        partial_blocks = real_partial.to(tl.int32).reshape(
            ratio, BLOCKS_PER_SEG, ratio
        )
        partial_block_mask = tl.sum(tl.sum(partial_blocks, 2), 0) > 0

        k_block_base = seg_start // ratio
        k_block_ids = k_block_base + offs_kb
        valid_store = q_block_valid & (k_block_ids < num_k_blocks)
        tl.store(
            p_out + k_block_ids,
            block_sums.to(Out.type.element_ty),
            mask=valid_store,
        )
        tl.store(
            p_out_mask + k_block_ids,
            partial_block_mask.to(tl.int8),
            mask=valid_store,
        )

    if is_causal:
        zero_vals = tl.zeros([BLOCKS_PER_SEG], dtype=tl.float32)
        for seg_idx in range(num_active_segments, num_segments):
            seg_start = seg_idx * SEGMENT_SIZE
            k_block_base = seg_start // ratio
            k_block_ids = k_block_base + offs_kb
            valid_store = q_block_valid & (k_block_ids < num_k_blocks)
            tl.store(
                p_out + k_block_ids,
                zero_vals.to(Out.type.element_ty),
                mask=valid_store,
            )
            tl.store(
                p_out_mask + k_block_ids,
                zero_mask,
                mask=valid_store,
            )


def _extract_raw_ptrs(
    startend_row_indices: paddle.Tensor,
    causal: bool,
) -> tuple[int, RawPtrs]:
    """
    startend_row_indices: [B, HIDS, seqlen_k, mode], mode in {1,2,4}
    - mode=1: only lt_start
    - mode=2:
        causal=True  -> (lt_start, lt_end)
        causal=False -> (lt_start, ut_end)
    - mode=4: (lt_start, lt_end, ut_start, ut_end)
    """
    mode = int(startend_row_indices.shape[-1])
    _require(mode in (1, 2, 4), f"Unsupported mode={mode}, expected 1/2/4")
    _require(
        not (causal and mode == 4),
        "mode=4 is only valid when causal=False in FlashMask semantics",
    )

    x = startend_row_indices.contiguous()
    lt_start = x[..., 0].contiguous()

    lt_end = lt_start
    ut_start = lt_start
    ut_end = lt_start

    if mode == 2:
        if causal:
            lt_end = x[..., 1].contiguous()
        else:
            ut_end = x[..., 1].contiguous()
    elif mode == 4:
        lt_end = x[..., 1].contiguous()
        ut_start = x[..., 2].contiguous()
        ut_end = x[..., 3].contiguous()

    return mode, RawPtrs(
        lt_start=lt_start,
        lt_end=lt_end,
        ut_start=ut_start,
        ut_end=ut_end,
    )


def _prepare_stride_maxmin_ptrs(
    raw: RawPtrs,
    mode: int,
    causal: bool,
    stride: int,
) -> StrideMaxMinPtrs:
    lt_start_max, lt_start_min = prepare_maxmin(raw.lt_start, stride)
    n_strides = int(lt_start_max.shape[2])

    dummy_max = lt_start_max
    lt_end_max = lt_end_min = dummy_max
    ut_start_max = ut_start_min = dummy_max
    ut_end_max = ut_end_min = dummy_max

    if mode == 2:
        if causal:
            lt_end_max, lt_end_min = prepare_maxmin(raw.lt_end, stride)
        else:
            ut_end_max, ut_end_min = prepare_maxmin(raw.ut_end, stride)
    elif mode == 4:
        lt_end_max, lt_end_min = prepare_maxmin(raw.lt_end, stride)
        ut_start_max, ut_start_min = prepare_maxmin(raw.ut_start, stride)
        ut_end_max, ut_end_min = prepare_maxmin(raw.ut_end, stride)

    return StrideMaxMinPtrs(
        lt_start_max=lt_start_max,
        lt_start_min=lt_start_min,
        lt_end_max=lt_end_max,
        lt_end_min=lt_end_min,
        ut_start_max=ut_start_max,
        ut_start_min=ut_start_min,
        ut_end_max=ut_end_max,
        ut_end_min=ut_end_min,
        n_strides=n_strides,
    )


def _is_trivial_nomask(
    startend_row_indices: paddle.Tensor,
    q_len: int,
    causal: bool,
) -> bool:
    mode = int(startend_row_indices.shape[-1])
    if causal:
        if mode != 1:
            return False
        return bool(paddle.all(startend_row_indices[..., 0] == q_len).item())

    if mode != 2:
        return False
    left_ok = paddle.all(startend_row_indices[..., 0] == q_len)
    right_ok = paddle.all(startend_row_indices[..., 1] == 0)
    return bool((left_ok & right_ok).item())


def _resolve_qchunk_config(
    *,
    stride: int,
    chunk_size: int,
    config: RRAttnConfig,
) -> tuple[int, int]:
    _require(BLOCK_SIZE % stride == 0, "stride must divide BLOCK_SIZE=128")
    _require(chunk_size > 0, "chunk_size must be positive")
    ratio = BLOCK_SIZE // stride

    chunk_q_strides = max(ratio, chunk_size // stride)
    chunk_q_strides = ((chunk_q_strides + ratio - 1) // ratio) * ratio

    _require(
        config.segment_size >= ratio and config.segment_size % ratio == 0,
        f"segment_size={config.segment_size} must be a positive multiple of ratio={ratio}",
    )

    return chunk_q_strides, config.segment_size


def _launch_qchunk_two_kernel(
    q: paddle.Tensor,
    k: paddle.Tensor,
    attn_sums: paddle.Tensor,
    boundary_protection_mask: paddle.Tensor,
    *,
    stride: int,
    causal: bool,
    scale: float,
    num_q_blocks: int,
    num_k_blocks: int,
    n_strides: int,
    chunk_size: int,
    config: RRAttnConfig,
    mask_ctx: MaskContext | None = None,
) -> None:
    bsz, q_len, num_q_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape

    n_q_strides = triton.cdiv(q_len, stride)
    n_k_strides = n_strides
    ratio = BLOCK_SIZE // stride
    gqa_groups = num_q_heads // num_kv_heads

    default_chunk_q_strides, segment_size = _resolve_qchunk_config(
        stride=stride,
        chunk_size=chunk_size,
        config=config,
    )
    max_chunk_q_strides = min(default_chunk_q_strides, n_q_strides)
    while segment_size > n_k_strides and segment_size > ratio:
        segment_size //= 2

    shift_tokens = kv_len - q_len

    intermediate = paddle.empty(
        (bsz, num_q_heads, max_chunk_q_strides, n_k_strides),
        dtype=q.dtype,
        device=q.device,
    )
    in_stride_b = num_q_heads * max_chunk_q_strides * n_k_strides
    in_stride_h = max_chunk_q_strides * n_k_strides
    in_stride_q = n_k_strides

    out_stride_b = num_q_heads * num_q_blocks * num_k_blocks
    out_stride_h = num_q_blocks * num_k_blocks
    out_stride_qb = num_k_blocks

    launch_kwargs = {
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }
    block_m = config.block_m
    block_n = config.block_n
    gqa_heads_per_cta = gqa_groups
    groups_per_kv = triton.cdiv(gqa_groups, gqa_heads_per_cta)

    for chunk_q_start in range(0, n_q_strides, max_chunk_q_strides):
        chunk_q_strides = min(max_chunk_q_strides, n_q_strides - chunk_q_start)

        grid_gemm = (
            triton.cdiv(chunk_q_strides, block_m),
            triton.cdiv(n_k_strides, block_n),
            bsz * num_kv_heads * groups_per_kv,
        )
        rrattn_gemm_qchunk_gqa_kernel[grid_gemm](
            q,
            k,
            intermediate,
            seqlen_q=q_len,
            seqlen_k=kv_len,
            chunk_q_start=chunk_q_start,
            chunk_q_strides=chunk_q_strides,
            n_k_strides=n_k_strides,
            shift_tokens=shift_tokens,
            out_stride_b=in_stride_b,
            out_stride_h=in_stride_h,
            out_stride_q=in_stride_q,
            HQ=num_q_heads,
            H=num_kv_heads,
            STRIDE=stride,
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            GQA_HEADS_PER_CTA=gqa_heads_per_cta,
            is_causal=causal,
            **launch_kwargs,
        )

        grid_output = (
            triton.cdiv(chunk_q_strides, ratio),
            num_q_heads,
            bsz,
        )
        if mask_ctx is None:
            rrattn_nomask_softmax_reduce_kernel[grid_output](
                intermediate,
                attn_sums,
                boundary_protection_mask,
                scale,
                in_stride_b=in_stride_b,
                in_stride_h=in_stride_h,
                in_stride_q=in_stride_q,
                out_stride_b=out_stride_b,
                out_stride_h=out_stride_h,
                out_stride_qb=out_stride_qb,
                chunk_q_start=chunk_q_start,
                chunk_q_strides=chunk_q_strides,
                n_k_strides=n_k_strides,
                num_q_blocks=num_q_blocks,
                num_k_blocks=num_k_blocks,
                seqlen_q=q_len,
                shift_tokens=shift_tokens,
                STRIDE=stride,
                ratio=ratio,
                SEGMENT_SIZE=segment_size,
                is_causal=causal,
                **launch_kwargs,
            )
        else:
            stride_mm = mask_ctx.stride_mm
            rrattn_flashmask_softmax_reduce_qchunk_kernel[grid_output](
                intermediate,
                attn_sums,
                boundary_protection_mask,
                stride_mm.lt_start_max,
                stride_mm.lt_start_min,
                stride_mm.lt_end_max,
                stride_mm.lt_end_min,
                stride_mm.ut_start_max,
                stride_mm.ut_start_min,
                stride_mm.ut_end_max,
                stride_mm.ut_end_min,
                scale,
                in_stride_b=in_stride_b,
                in_stride_h=in_stride_h,
                in_stride_q=in_stride_q,
                out_stride_b=out_stride_b,
                out_stride_h=out_stride_h,
                out_stride_qb=out_stride_qb,
                chunk_q_start=chunk_q_start,
                chunk_q_strides=chunk_q_strides,
                n_k_strides=n_k_strides,
                num_q_blocks=num_q_blocks,
                num_k_blocks=num_k_blocks,
                seqlen_q=q_len,
                shift_tokens=shift_tokens,
                HQ=num_q_heads,
                HIDS=mask_ctx.num_indices_heads,
                STRIDE=stride,
                ratio=ratio,
                SEGMENT_SIZE=segment_size,
                mode=mask_ctx.mode,
                is_causal=causal,
                **launch_kwargs,
            )


def rr_attn_estimate_triton_func(
    q: paddle.Tensor,
    k: paddle.Tensor,
    startend_row_indices: paddle.Tensor,
    *,
    stride: int = 8,
    causal: bool = True,
    threshold: float = 1.0,
    chunk_size: int = 16384,
    config: RRAttnConfig | None = None,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    q: [B, seqlen_q, Hq, D]
    k: [B, seqlen_k, Hkv, D]
    startend_row_indices: [B, Hids, seqlen_k, mode]

    Returns:
      attn_sums: [B, Hq, ceil(seqlen_q/128), ceil(seqlen_k/128)]
      boundary_protection_mask: same shape, bool
      selected_blocks: same shape, bool
    """
    _require(stride > 0, "stride must be positive")
    _require(BLOCK_SIZE % stride == 0, "stride must divide BLOCK_SIZE=128")
    _require(chunk_size > 0, "chunk_size must be positive")

    bsz, q_len, num_q_heads, head_dim = q.shape
    bsz2, kv_len, num_kv_heads, head_dim_k = k.shape
    _require(bsz2 == bsz, "q/k batch size mismatch")
    _require(head_dim_k == head_dim, "q/k head_dim mismatch")
    _require(
        startend_row_indices.shape[0] == bsz,
        "startend_row_indices batch mismatch",
    )
    _require(
        startend_row_indices.shape[2] == kv_len,
        "startend_row_indices seqlen_k mismatch",
    )
    _require(
        num_q_heads % num_kv_heads == 0,
        "MHA/GQA requires num_q_heads % num_kv_heads == 0",
    )
    config = _normalize_config(config, head_dim=head_dim)

    num_indices_heads = int(startend_row_indices.shape[1])
    _require(
        num_q_heads % num_indices_heads == 0,
        "Require num_q_heads % num_indices_heads == 0 for head mapping",
    )

    num_q_blocks = triton.cdiv(q_len, BLOCK_SIZE)
    num_k_blocks = triton.cdiv(kv_len, BLOCK_SIZE)
    attn_sums = paddle.empty(
        (bsz, num_q_heads, num_q_blocks, num_k_blocks),
        dtype=q.dtype,
        device=q.device,
    )
    boundary_protection_mask = paddle.empty(
        (bsz, num_q_heads, num_q_blocks, num_k_blocks),
        dtype=paddle.int8,
        device=q.device,
    )

    scale = LOG2E / math.sqrt(head_dim) / stride
    with use_compat_guard(silent=True):
        if _is_trivial_nomask(startend_row_indices, q_len, causal):
            mask_ctx = None
            n_strides = triton.cdiv(kv_len, stride)
        else:
            mode, raw = _extract_raw_ptrs(startend_row_indices, causal)
            stride_mm = _prepare_stride_maxmin_ptrs(raw, mode, causal, stride)
            mask_ctx = MaskContext(
                mode=mode,
                stride_mm=stride_mm,
                num_indices_heads=num_indices_heads,
            )
            n_strides = stride_mm.n_strides

        _launch_qchunk_two_kernel(
            q,
            k,
            attn_sums,
            boundary_protection_mask,
            stride=stride,
            causal=causal,
            scale=scale,
            num_q_blocks=num_q_blocks,
            num_k_blocks=num_k_blocks,
            n_strides=n_strides,
            chunk_size=chunk_size,
            config=config,
            mask_ctx=mask_ctx,
        )

        selected_blocks = find_blocks_topp(
            attn_sums.astype(paddle.float32), float(threshold)
        )
    if causal:
        visible_mask = _build_fa3_causal_block_visible_mask(
            attn_sums, q_len, kv_len
        )
        selected_blocks = paddle.logical_and(selected_blocks, visible_mask)
        mandatory_mask = _build_causal_prefill_mandatory_mask(
            attn_sums, q_len, kv_len
        )
        selected_blocks = paddle.logical_or(selected_blocks, mandatory_mask)
    boundary_protection_mask = boundary_protection_mask.astype(paddle.bool)
    selected_blocks = paddle.logical_or(
        selected_blocks, boundary_protection_mask
    )
    return attn_sums, boundary_protection_mask, selected_blocks


enable_profile = False
attn_time_ms = 0.0
estimate_func_time_ms = 0.0


def set_profile(enable=True):
    global enable_profile
    enable_profile = enable


def is_enable_profile():
    global enable_profile
    return enable_profile


def set_attn_time(attn_time=0.0):
    global attn_time_ms
    attn_time_ms = attn_time


def get_attn_time():
    global attn_time_ms
    return attn_time_ms


def add_attn_time(attn_time):
    global attn_time_ms
    attn_time_ms += attn_time


def set_estimate_func_time(time_ms=0.0):
    global estimate_func_time_ms
    estimate_func_time_ms = time_ms


def get_estimate_func_time():
    global estimate_func_time_ms
    return estimate_func_time_ms


def add_estimate_func_time(time_ms):
    global estimate_func_time_ms
    estimate_func_time_ms += time_ms


def can_use_triton_kernels():
    if not paddle.device.is_compiled_with_cuda():
        return False
    try:
        return paddle.device.get_device().startswith("gpu")
    except Exception:
        return False


def _compute_sparse_ratio(
    block_mask: paddle.Tensor,
    *,
    q_block_num: int,
    k_block_num: int,
    num_heads: int,
    causal: bool,
):
    if causal:
        offset = k_block_num - q_block_num
        visible_blocks = 0
        for q_idx in range(q_block_num):
            visible_blocks += min(k_block_num, max(0, q_idx + offset + 1))
        num_to_compute = visible_blocks * num_heads
    else:
        num_to_compute = q_block_num * k_block_num * num_heads
    sparse_ratio = 1.0 - (
        block_mask.astype(paddle.float32).sum()
        / max(float(num_to_compute), 1.0)
    )
    return paddle.clip(sparse_ratio, min=0.0, max=1.0)


def _build_nomask_startend(
    *,
    batch_size: int,
    q_len: int,
    k_len: int,
    causal: bool,
    device,
) -> paddle.Tensor:
    if causal:
        return paddle.full(
            (batch_size, 1, k_len, 1),
            q_len,
            dtype=paddle.int32,
            device=device,
        )

    start = paddle.full(
        (batch_size, 1, k_len, 1),
        q_len,
        dtype=paddle.int32,
        device=device,
    )
    end = paddle.zeros(
        (batch_size, 1, k_len, 1), dtype=paddle.int32, device=device
    )
    return paddle.concat([start, end], axis=-1)


def _block_sparse_attention_blsd(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    block_mask: paddle.Tensor,
    startend_row_indices: paddle.Tensor | None = None,
    block_size: int = 128,
    causal: bool = True,
):
    batch_size, q_len, num_q_heads, head_dim = query_states.shape
    _, k_len, num_kv_heads, _ = key_states.shape

    assert block_size == 128, (
        "F.flashmask_attention block_mask only supports block_size=128"
    )
    assert head_dim == 128, (
        "F.flashmask_attention block_mask only supports head_dim=128"
    )
    assert num_q_heads % num_kv_heads == 0, (
        "MHA/GQA requires num_q_heads % num_kv_heads == 0"
    )
    assert value_states.shape[1] == k_len, "key/value sequence length mismatch"
    assert value_states.shape[2] == num_kv_heads, (
        "key/value head count mismatch"
    )
    assert block_mask.shape[1] == num_q_heads, (
        "block_mask head count must match query heads"
    )

    block_mask = block_mask.astype(paddle.int32).contiguous()
    if startend_row_indices is None:
        startend_row_indices = _build_nomask_startend(
            batch_size=batch_size,
            q_len=q_len,
            k_len=k_len,
            causal=causal,
            device=query_states.device,
        )
    if startend_row_indices.shape[1] != block_mask.shape[1]:
        num_indices_heads = int(startend_row_indices.shape[1])
        assert block_mask.shape[1] % num_indices_heads == 0, (
            "block_mask heads must be divisible by startend heads"
        )
        startend_row_indices = startend_row_indices.repeat_interleave(
            block_mask.shape[1] // num_indices_heads,
            axis=1,
        )
    startend_row_indices = startend_row_indices.contiguous()

    attn_output = F.flashmask_attention(
        query_states,
        key_states,
        value_states,
        startend_row_indices=startend_row_indices,
        dropout=0.0,
        causal=causal,
        block_mask=block_mask,
    )
    return attn_output.contiguous()


def block_sparse_attention(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    block_mask: paddle.Tensor,
    startend_row_indices: paddle.Tensor | None = None,
    block_size: int = 128,
    causal: bool = True,
):
    return _block_sparse_attention_blsd(
        query_states.transpose(1, 2).contiguous(),
        key_states.transpose(1, 2).contiguous(),
        value_states.transpose(1, 2).contiguous(),
        block_mask,
        startend_row_indices=startend_row_indices,
        block_size=block_size,
        causal=causal,
    )


def _rrattn_estimate_blsd(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    block_size=128,
    stride=8,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    layer_idx=None,
    startend_row_indices=None,
    config: RRAttnConfig | None = None,
    **kwargs,
):
    del norm, softmax, select_mode, layer_idx, kwargs

    if not use_triton:
        raise NotImplementedError("rrattn currently requires use_triton=True")
    if not can_use_triton_kernels():
        raise RuntimeError("rrattn Triton kernels require a CUDA gpu device")
    if kdb != 1:
        raise ValueError("rrattn Triton kernels require kdb=1")
    assert block_size == 128, "RRAttention currently requires block_size=128"
    assert stride > 0, "stride must be positive"
    assert block_size % stride == 0, "stride must divide block_size=128"
    assert chunk_size is not None and chunk_size > 0, (
        "chunk_size must be positive"
    )

    batch_size, q_len, num_q_heads, _ = query_states.shape
    _, k_len, num_kv_heads, _ = key_states.shape
    assert num_q_heads % num_kv_heads == 0, (
        "MHA/GQA requires num_q_heads % num_kv_heads == 0"
    )

    if key_states.device != query_states.device:
        key_states = key_states.to(query_states.device)

    if startend_row_indices is None:
        startend_row_indices = _build_nomask_startend(
            batch_size=batch_size,
            q_len=q_len,
            k_len=k_len,
            causal=causal,
            device=query_states.device,
        )

    attn_sums, _, selected_blocks = rr_attn_estimate_triton_func(
        query_states,
        key_states,
        startend_row_indices,
        stride=stride,
        causal=causal,
        threshold=threshold,
        chunk_size=chunk_size,
        config=config,
    )

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    assert attn_sums.shape[2] == q_block_num, "estimate q-block count mismatch"
    assert attn_sums.shape[3] == k_block_num, "estimate k-block count mismatch"
    block_mask = selected_blocks

    if keep_sink and k_block_num > 0:
        block_mask[:, :, :, 0] = True
    if keep_recent and q_block_num > 0:
        block_mask[:, :, -1, :k_block_num] = True

    return attn_sums, block_mask


def rrattn_estimate(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    block_size=128,
    stride=8,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
    rrattn_version="v1",
    layer_idx=None,
    startend_row_indices=None,
    config: RRAttnConfig | None = None,
    **kwargs,
):
    return _rrattn_estimate_blsd(
        query_states.transpose(1, 2).contiguous(),
        key_states.transpose(1, 2).contiguous(),
        block_size=block_size,
        stride=stride,
        norm=norm,
        softmax=softmax,
        threshold=threshold,
        chunk_size=chunk_size,
        select_mode=select_mode,
        use_triton=use_triton,
        causal=causal,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        layer_idx=layer_idx,
        startend_row_indices=startend_row_indices,
        config=config,
        **kwargs,
    )


def rrattn_prefill(
    query_states: paddle.Tensor,
    key_states: paddle.Tensor,
    value_states: paddle.Tensor,
    stride=8,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
    rrattn_version="v1",
    layer_idx=None,
    startend_row_indices=None,
    config=None,
):
    batch_size, num_q_heads, q_len, _ = query_states.shape
    _, num_kv_heads, k_len, _ = key_states.shape
    assert num_q_heads % num_kv_heads == 0, (
        "MHA/GQA requires num_q_heads % num_kv_heads == 0"
    )
    assert value_states.shape[1] == num_kv_heads, (
        "key/value head count mismatch"
    )
    assert value_states.shape[2] == k_len, "key/value sequence length mismatch"

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            )
        )
    chunk_size = min(
        (q_len + (block_size * stride) - 1)
        // (block_size * stride)
        * (block_size * stride),
        chunk_size,
    )
    if key_states.device != query_states.device:
        key_states = key_states.to(query_states.device)
    if value_states.device != query_states.device:
        value_states = value_states.to(query_states.device)
    if startend_row_indices is None:
        rrattn_startend_row_indices = _build_nomask_startend(
            batch_size=batch_size,
            q_len=q_len,
            k_len=k_len,
            causal=causal,
            device=query_states.device,
        )
    else:
        rrattn_startend_row_indices = startend_row_indices
    query_states_blsd = query_states.transpose(1, 2).contiguous()
    key_states_blsd = key_states.transpose(1, 2).contiguous()
    value_states_blsd = value_states.transpose(1, 2).contiguous()

    if is_enable_profile():
        paddle.cuda.synchronize()
        start_event = paddle.cuda.Event(enable_timing=True)
        end_event = paddle.cuda.Event(enable_timing=True)
        start_event.record()

    attn_sums, approx_simple_mask = _rrattn_estimate_blsd(
        query_states_blsd,
        key_states_blsd,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=use_triton,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
        layer_idx=layer_idx,
        startend_row_indices=rrattn_startend_row_indices,
        config=config,
    )
    if is_enable_profile():
        end_event.record()
        paddle.cuda.synchronize()
        add_estimate_func_time(start_event.elapsed_time(end_event))

    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    if (
        approx_simple_mask.shape[2] != q_block_num
        or approx_simple_mask.shape[3] != k_block_num
    ):
        approx_simple_mask = approx_simple_mask[
            :, :, :q_block_num, :k_block_num
        ].contiguous()
    sparse_ratio = _compute_sparse_ratio(
        approx_simple_mask,
        q_block_num=q_block_num,
        k_block_num=k_block_num,
        num_heads=num_q_heads,
        causal=causal,
    )

    if is_enable_profile():
        paddle.cuda.synchronize()
        start_event = paddle.cuda.Event(enable_timing=True)
        end_event = paddle.cuda.Event(enable_timing=True)
        start_event.record()
    attn_output = _block_sparse_attention_blsd(
        query_states_blsd,
        key_states_blsd,
        value_states_blsd,
        approx_simple_mask,
        startend_row_indices=rrattn_startend_row_indices,
        block_size=block_size,
        causal=causal,
    )
    if is_enable_profile():
        end_event.record()
        paddle.cuda.synchronize()
        add_attn_time(start_event.elapsed_time(end_event))

    del query_states
    del approx_simple_mask, attn_sums
    return attn_output, sparse_ratio


__all__ = [
    "RRAttnConfig",
    "get_rrattn_config",
    "rrattn_estimate",
    "rrattn_prefill",
    "can_use_triton_kernels",
    "set_profile",
    "is_enable_profile",
    "set_attn_time",
    "get_attn_time",
    "add_attn_time",
    "set_estimate_func_time",
    "get_estimate_func_time",
    "add_estimate_func_time",
]
