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

"""CTA block-scan helpers adapted from cuDNN Frontend.

The implementation follows the DSA indexer ``block_scan.py`` reference:
warp-level inclusive scans use shuffle instructions, warp totals are reduced
in shared memory, and a CTA barrier completes the cross-warp scan.  The
standalone PaddleFleet operator uses the helper for its 256-bin radix
histograms.
"""

from __future__ import annotations

import math

import cutlass
from cutlass import cute
from cutlass._mlir.dialects import llvm
from cutlass.utils.smem_allocator import SmemAllocator


@cute.jit
def fence_acq_rel_cta(*, loc=None, ip=None):
    """Make CTA-scoped global/shared writes visible before a barrier."""
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="membar.cta;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def warp_scan(
    val: cutlass.Int32,
    tidx,
    lane_id,
    num_threads_per_warp: cutlass.Constexpr,
):
    """Inclusive scan within one warp or a power-of-two warp subgroup."""
    mask_val = cutlass.const_expr(
        ((1 << num_threads_per_warp) - 1) & 0xFFFFFFFF
    )
    mask_and_clamp_val = 0
    iteration = cute.arch.log2_of_pow2_int(cutlass.Int32(num_threads_per_warp))
    for i in cutlass.range(iteration, unroll_full=True):
        offset = 1 << i
        other = cute.arch.shuffle_sync_up(
            val,
            offset,
            mask=mask_val,
            mask_and_clamp=mask_and_clamp_val,
        )
        if lane_id >= offset:
            val = val + other
    return val


@cute.jit
def block_prefix_sum_kernel(
    val: cutlass.Int32,
    warp_sums: cute.Tensor,
    tidx,
    num_threads,
    num_warps,
    barrier_id=1,
    need_total_sum=False,
):
    """Compute an inclusive block prefix sum using warp shuffles."""
    warp_id = tidx // 32
    lane_id = tidx % 32

    assert num_threads % 32 == 0
    assert num_warps > 1
    assert num_warps == 2 ** int(math.log2(num_warps))

    val = warp_scan(val, tidx, lane_id, num_threads_per_warp=32)
    if lane_id == 31:
        warp_sums[warp_id] = val
    cute.arch.barrier(
        barrier_id=barrier_id,
        number_of_threads=num_threads,
    )

    if warp_id == 0:
        if lane_id < num_warps:
            warp_val = warp_sums[lane_id]
            warp_val = warp_scan(
                warp_val,
                tidx,
                lane_id,
                num_threads_per_warp=num_warps,
            )
            warp_sums[lane_id] = warp_val
    cute.arch.barrier(
        barrier_id=barrier_id,
        number_of_threads=num_threads,
    )

    if warp_id > 0:
        val = val + warp_sums[warp_id - 1]

    total_sum = 0
    if need_total_sum:
        total_sum = warp_sums[num_warps - 1]
    return val, total_sum


@cute.kernel
def block_prefix_sum(
    num_bins: cutlass.Constexpr,
    num_threads_per_block: cutlass.Constexpr,
    input: cute.Tensor,
    output: cute.Tensor,
):
    """Standalone block-prefix-sum kernel from the reference implementation."""
    tidx, _, _ = cute.arch.thread_idx()
    num_warps = cutlass.const_expr(min(num_bins, num_threads_per_block) // 32)
    smem = SmemAllocator()
    warp_sums = smem.allocate_tensor(
        element_type=cutlass.Int32,
        layout=cute.make_ordered_layout((num_warps,), order=(0,)),
        byte_alignment=128,
    )

    if cutlass.const_expr(num_bins < num_threads_per_block):
        if tidx < num_bins:
            val = input[tidx]
            val, _ = block_prefix_sum_kernel(
                val,
                warp_sums,
                tidx,
                num_bins,
                num_warps,
                barrier_id=1,
            )
            output[tidx] = val
    elif cutlass.const_expr(num_bins == num_threads_per_block):
        val = input[tidx]
        val, _ = block_prefix_sum_kernel(
            val,
            warp_sums,
            tidx,
            num_bins,
            num_warps,
            barrier_id=1,
        )
        output[tidx] = val
    else:
        assert num_bins % num_threads_per_block == 0
        previous_sum = 0
        val = 0
        total_sum = 0
        for i in range(tidx, num_bins, num_threads_per_block):
            val = input[i]
            val, total_sum = block_prefix_sum_kernel(
                val,
                warp_sums,
                tidx,
                num_threads_per_block,
                num_warps,
                barrier_id=0,
                need_total_sum=True,
            )
            output[i] = val + previous_sum
            previous_sum = previous_sum + total_sum
