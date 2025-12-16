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

import triton
import triton.language as tl


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
def _is_block_fully_masked(
    block_rows,
    lts_max,
    lte_min,
    uts_max,
    ute_min,
):
    # since we pass exact row indices now, use "<" for end
    in_lt = (block_rows[:, None] >= lts_max[None, :]) & (
        block_rows[:, None] < lte_min[None, :]
    )
    in_ut = (block_rows[:, None] >= uts_max[None, :]) & (
        block_rows[:, None] < ute_min[None, :]
    )

    mask = in_lt | in_ut
    return mask


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

    fm_geo = _is_block_fully_masked(
        q_rows,
        fm_lts,
        fm_lte,
        fm_uts,
        fm_ute,
    )
    fm_oob = ~k_load_mask[None, :]

    return fm_geo | fm_oob


@triton.jit
def _is_block_partially_masked(
    block_rows,
    lts_min,
    lte_max,
    uts_min,
    ute_max,
):
    # Logic: Overlap exists if Q is potentially inside [min_start, max_end)
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
