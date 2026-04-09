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
- flashmask v3 (SM90/SM100 era) and computes per-
  block (kBlockN) maxima and minima across a 1D input sequence then stores
  them in output buffers laid out either continuously or in aligned "chunks".
"""

from typing import Optional, NamedTuple
from dataclasses import dataclass
import paddle
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
from flash_mask.cute import utils
import operator

__all__ = [
    "prepare_block_maxmin",
    "FlashMaskInfoPaddle"
]


# Keep the same constant name for clarity with the CUDA source.
flashmask_buffer_length = 16 * 1024


class FlashMaskInfo(NamedTuple):
    is_causal: bool
    startend_row_indices: cute.Tensor
    LTS_nblock_max: Optional[cute.Tensor]
    LTS_nblock_min: Optional[cute.Tensor]
    LTE_nblock_max: Optional[cute.Tensor]
    LTE_nblock_min: Optional[cute.Tensor]
    UTS_nblock_max: Optional[cute.Tensor]
    UTS_nblock_min: Optional[cute.Tensor]
    UTE_nblock_max: Optional[cute.Tensor]
    UTE_nblock_min: Optional[cute.Tensor]
    valid_block_count: Optional[cute.Tensor]

    def __new_from_mlir_values__(self, values):
        if len(values) == 3:
            values = (self.is_causal, *values, None, None, None, None, None, None, None)
        elif len(values) == 4:
            values = (self.is_causal, *values[:3], None, None, None, None, None, None, *values[3:])
        elif self.is_causal and len(values) == 5:
            values = (self.is_causal, *values, None, None, None, None, None)
        elif self.is_causal and len(values) == 6:
            values = (self.is_causal, *values[:5], None, None, None, None, *values[5:])
        elif not self.is_causal and len(values) == 5:
            values = (self.is_causal, *values[:3], None, None, None, None, *values[3:], None)
        elif not self.is_causal and len(values) == 6:
            values = (self.is_causal, *values[:3], None, None, None, None, *values[3:5], *values[5:])
        elif len(values) == 9:
            values = (self.is_causal, *values, None)
        else:
            values = (self.is_causal, *values)
        return FlashMaskInfo(*values)


@dataclass
class FlashMaskInfoPaddle:
    is_causal: bool
    startend_row_indices: paddle.Tensor
    LTS_nblock_max: Optional[paddle.Tensor] = None
    LTS_nblock_min: Optional[paddle.Tensor] = None
    LTE_nblock_max: Optional[paddle.Tensor] = None
    LTE_nblock_min: Optional[paddle.Tensor] = None
    UTS_nblock_max: Optional[paddle.Tensor] = None
    UTS_nblock_min: Optional[paddle.Tensor] = None
    UTE_nblock_max: Optional[paddle.Tensor] = None
    UTE_nblock_min: Optional[paddle.Tensor] = None
    valid_block_count: Optional[paddle.Tensor] = None


def _compute_nblock_seqlen(seqlen_k: int, kBlockN: int) -> int:
    """Compute the padded number of blocks (the same formula as original).

    Uses: ((n + kBlockN - 1) / kBlockN + 3) & 0xfffffffc
    The +3 then & ~3 is to make padding to a multiple of 4 (umising: int4 load)
    """
    nblock = (seqlen_k + kBlockN - 1) // kBlockN
    return (nblock + 3) & 0xfffffffc


@cute.kernel
def scan_max_min_kernel(
    mInput: cute.Tensor,  # expected shape (b, n)
    b: cutlass.Int32,
    n: cutlass.Int32,
    kBlockN: cutlass.Int32,
    mMaxO: cute.Tensor,
    mMinO: cute.Tensor,
):
    # thread / block indices
    tidx, tidy, tidz = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    bd_x, bd_y, bd_z = cute.arch.block_dim()

    nblock = (n + kBlockN - 1) // kBlockN
    nblock_seqlen = ((nblock + 3) // 4) * 4

    mInput = cute.make_tensor(mInput.iterator, cute.make_layout((cutlass.Int32(b), cutlass.Int32(n)), stride=(cutlass.Int32(n), cutlass.Int32(1))))
    mMaxO = cute.make_tensor(mMaxO.iterator, cute.make_layout((cutlass.Int32(b * nblock_seqlen)), stride=(cutlass.Int32(1))))
    mMinO = cute.make_tensor(mMinO.iterator, cute.make_layout((cutlass.Int32(b * nblock_seqlen)), stride=(cutlass.Int32(1))))

    # compute batch row id
    bid_row = tidy + bidy * bd_y
    if bid_row < b:
        nblock_idx = bidx

        # lane id within warp
        lane_id = tidx % cute.arch.WARP_SIZE

        # number of 32-element strides per thread required to cover kBlockN
        nums = (kBlockN + 31) // 32

        # Per-thread partial max/min initial values
        maxv = cutlass.Int32(0)
        minv = cutlass.Int32(0x7FFFFFFF)

        # Per-thread starting global element index
        idx = nblock_idx * kBlockN + tidx
        for i in cutlass.range(nums, unroll=1):
            local_pos = lane_id + i * cute.arch.WARP_SIZE
            if (local_pos < kBlockN) and (idx < n):
                # load element (mInput is (b, n))
                # Use domain_offset-style indexing: mInput[bid_row, idx]
                val = cutlass.Int32(mInput[bid_row, idx])
                # update per-thread min/max
                maxv = cutlass.max(maxv, val)
                minv = cutlass.min(minv, val)
            idx = idx + 32

        cute.arch.sync_warp()
        # Warp-level reduction: reduce across the 32 lanes in the warp
        warp_max = utils.warp_reduce(maxv, lambda x, y: cutlass.max(x, y), width=cute.arch.WARP_SIZE)
        warp_min = utils.warp_reduce(minv, lambda x, y: cutlass.min(x, y), width=cute.arch.WARP_SIZE)

        # lane 0 writes the reduced result (one writer per warp)
        if lane_id == 0:
            # compute storage layout indexes similar to CUDA code
            # nblock_seqlen = ((n + kBlockN - 1) / kBlockN + 3) & 0xfffffffc  --> round up to multiple of 4

            dest_idx = bid_row * nblock_seqlen + nblock_idx
            #cute.printf(tidx, tidy, nblock, nblock_seqlen, dest_idx)

            # store to output tensors
            mMaxO[dest_idx] = warp_max
            mMinO[dest_idx] = warp_min

@cute.jit
def scan_max_min_cute(
    mInput: cute.Tensor,  # expected shape (b, h, s)
    b: cutlass.Int32,
    n: cutlass.Int32,
    kBlockN: cutlass.Int32,
    stream: cuda.CUstream,
    mMaxO: cute.Tensor,
    mMinO: cute.Tensor,
):
    scan_max_min_kernel(
        mInput,
        b, n, kBlockN,
        mMaxO,
        mMinO,
    ).launch(
        grid=[(n + kBlockN - 1) // kBlockN, (b + 3) // 4, cutlass.Int32(1)],
        block=[cutlass.Int32(32), cutlass.Int32(4), cutlass.Int32(1)],
        stream=stream,
    )

def _scan_max_min(
    mInput: paddle.Tensor,
    b: int,
    n: int,
    mMaxO: paddle.Tensor,
    mMinO: paddle.Tensor,
    kBlockN: int,
):
    input_tensor = from_dlpack(mInput.contiguous(), assumed_align=4).mark_layout_dynamic(leading_dim=2)
    max_tensor = from_dlpack(mMaxO, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    min_tensor = from_dlpack(mMinO, assumed_align=4).mark_layout_dynamic(leading_dim=2)

    current_stream = cuda.CUstream(paddle.device.current_stream().stream_base.cuda_stream)

    compile_key = (kBlockN,)
    if compile_key not in _scan_max_min.compile_cache:
        _scan_max_min.compile_cache[compile_key] = cute.compile(
            scan_max_min_cute,
            input_tensor,
            cutlass.Int32(b), cutlass.Int32(n), cutlass.Int32(kBlockN),
            current_stream,
            max_tensor,
            min_tensor,
        )
    _scan_max_min.compile_cache[compile_key](
        input_tensor,
        cutlass.Int32(b), cutlass.Int32(n), cutlass.Int32(kBlockN),
        current_stream,
        max_tensor,
        min_tensor,
    )

_scan_max_min.compile_cache = {}

def prepare_block_maxmin(flashmask_info: FlashMaskInfoPaddle, kBlockN: int = 128):
    """Prepare block-sparse max/min tensors for flashmask.

    The function will compute derived pointers/offsets and call scanMaxMinGpu
    for each existing input pointer.
    """

    batch, heads, seqlen_k, num_vecs = flashmask_info.startend_row_indices.shape
    nblocks = _compute_nblock_seqlen(seqlen_k, kBlockN)

    if num_vecs == 1 and flashmask_info.LTS_nblock_max is None and flashmask_info.LTS_nblock_min is None:
        flashmask_info.LTS_nblock_max = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.LTS_nblock_min = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        _scan_max_min(flashmask_info.startend_row_indices[..., 0], batch * heads, seqlen_k, flashmask_info.LTS_nblock_max, flashmask_info.LTS_nblock_min, kBlockN)
    elif num_vecs == 2 and flashmask_info.is_causal and (
            flashmask_info.LTS_nblock_max is None and flashmask_info.LTS_nblock_min is None and
            flashmask_info.LTE_nblock_max is None and flashmask_info.LTE_nblock_min is None
    ):
        flashmask_info.LTS_nblock_max = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.LTS_nblock_min = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.LTE_nblock_max = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.LTE_nblock_min = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        _scan_max_min(flashmask_info.startend_row_indices[..., 0], batch * heads, seqlen_k, flashmask_info.LTS_nblock_max, flashmask_info.LTS_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 1], batch * heads, seqlen_k, flashmask_info.LTE_nblock_max, flashmask_info.LTE_nblock_min, kBlockN)
    elif num_vecs == 2 and not flashmask_info.is_causal and (
            flashmask_info.LTS_nblock_max is None and flashmask_info.LTS_nblock_min is None and
            flashmask_info.UTE_nblock_max is None and flashmask_info.UTE_nblock_min is None
    ):
        flashmask_info.LTS_nblock_max = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.LTS_nblock_min = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.UTE_nblock_max = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.UTE_nblock_min = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        _scan_max_min(flashmask_info.startend_row_indices[..., 0], batch * heads, seqlen_k, flashmask_info.LTS_nblock_max, flashmask_info.LTS_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 1], batch * heads, seqlen_k, flashmask_info.UTE_nblock_max, flashmask_info.UTE_nblock_min, kBlockN)
    elif num_vecs == 4 and (
            flashmask_info.LTS_nblock_max is None and flashmask_info.LTS_nblock_min is None and
            flashmask_info.LTE_nblock_max is None and flashmask_info.LTE_nblock_min is None and
            flashmask_info.UTS_nblock_max is None and flashmask_info.UTS_nblock_min is None and
            flashmask_info.UTE_nblock_max is None and flashmask_info.UTE_nblock_min is None
            
    ):
        flashmask_info.LTS_nblock_max = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.LTS_nblock_min = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.LTE_nblock_max = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.LTE_nblock_min = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.UTS_nblock_max = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.UTS_nblock_min = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.UTE_nblock_max = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        flashmask_info.UTE_nblock_min = paddle.zeros([batch, heads, nblocks], dtype=paddle.int32)
        _scan_max_min(flashmask_info.startend_row_indices[..., 0], batch * heads, seqlen_k, flashmask_info.LTS_nblock_max, flashmask_info.LTS_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 1], batch * heads, seqlen_k, flashmask_info.LTE_nblock_max, flashmask_info.LTE_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 2], batch * heads, seqlen_k, flashmask_info.UTS_nblock_max, flashmask_info.UTS_nblock_min, kBlockN)
        _scan_max_min(flashmask_info.startend_row_indices[..., 3], batch * heads, seqlen_k, flashmask_info.UTE_nblock_max, flashmask_info.UTE_nblock_min, kBlockN)
    else:
        raise ValueError(f"Unsupported num_vecs={num_vecs} in flashmask_info")
    

def is_flashmask_enabled(flashmask_info: FlashMaskInfoPaddle) -> bool:
    return any(t is not None for t in (
        flashmask_info.LTS_nblock_max,
        flashmask_info.LTS_nblock_min,
        flashmask_info.LTE_nblock_max,
        flashmask_info.LTE_nblock_min,
        flashmask_info.UTS_nblock_max,
        flashmask_info.UTS_nblock_min,
        flashmask_info.UTE_nblock_max,
        flashmask_info.UTE_nblock_min,
    ))

def to_cute_flashmask_info(flashmask_info: FlashMaskInfoPaddle) -> Optional[FlashMaskInfo]:
    if not is_flashmask_enabled(flashmask_info):
        return None

    batch, heads, seqlen_k, num_vecs = flashmask_info.startend_row_indices.shape

    startend_row_indices_tensor = from_dlpack(flashmask_info.startend_row_indices, assumed_align=4).mark_layout_dynamic(leading_dim=3)
    LTS_nblock_max_tensor = None
    LTS_nblock_min_tensor = None
    LTE_nblock_max_tensor = None
    LTE_nblock_min_tensor = None
    UTS_nblock_max_tensor = None
    UTS_nblock_min_tensor = None
    UTE_nblock_max_tensor = None
    UTE_nblock_min_tensor = None

    if num_vecs == 1:
        LTS_nblock_max_tensor = from_dlpack(flashmask_info.LTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTS_nblock_min_tensor = from_dlpack(flashmask_info.LTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    elif num_vecs == 2 and flashmask_info.is_causal:
        LTS_nblock_max_tensor = from_dlpack(flashmask_info.LTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTS_nblock_min_tensor = from_dlpack(flashmask_info.LTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTE_nblock_max_tensor = from_dlpack(flashmask_info.LTE_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTE_nblock_min_tensor = from_dlpack(flashmask_info.LTE_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    elif num_vecs == 2 and not flashmask_info.is_causal:
        LTS_nblock_max_tensor = from_dlpack(flashmask_info.LTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTS_nblock_min_tensor = from_dlpack(flashmask_info.LTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTE_nblock_max_tensor = from_dlpack(flashmask_info.UTE_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTE_nblock_min_tensor = from_dlpack(flashmask_info.UTE_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    elif num_vecs == 4:
        LTS_nblock_max_tensor = from_dlpack(flashmask_info.LTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTS_nblock_min_tensor = from_dlpack(flashmask_info.LTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTE_nblock_max_tensor = from_dlpack(flashmask_info.LTE_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        LTE_nblock_min_tensor = from_dlpack(flashmask_info.LTE_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTS_nblock_max_tensor = from_dlpack(flashmask_info.UTS_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTS_nblock_min_tensor = from_dlpack(flashmask_info.UTS_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTE_nblock_max_tensor = from_dlpack(flashmask_info.UTE_nblock_max, assumed_align=4).mark_layout_dynamic(leading_dim=2)
        UTE_nblock_min_tensor = from_dlpack(flashmask_info.UTE_nblock_min, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    else:
        raise ValueError(f"Unsupported num_vecs={num_vecs} in flashmask_info")

    if flashmask_info.valid_block_count is not None:
        valid_block_count = from_dlpack(flashmask_info.valid_block_count, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    else:
        valid_block_count = None

    return FlashMaskInfo(
        flashmask_info.is_causal,
        startend_row_indices_tensor,
        LTS_nblock_max_tensor,
        LTS_nblock_min_tensor,
        LTE_nblock_max_tensor,
        LTE_nblock_min_tensor,
        UTS_nblock_max_tensor,
        UTS_nblock_min_tensor,
        UTE_nblock_max_tensor,
        UTE_nblock_min_tensor,
        valid_block_count
    )

@cute.kernel
def reduce_block_count_kernel(
    LTS_nblock_max: cute.Tensor, # [b, h, sk/kBlockN]
    LTE_nblock_min: cute.Tensor,
    UTS_nblock_max: cute.Tensor,
    UTE_nblock_min: cute.Tensor,
    valid_block_count: cute.Tensor, # [b, h, sQ/kBlockM] valid_block_count means how many block are not fully masked in each row
    num_blocks_row: cutlass.Int32,
    num_blocks_col: cutlass.Int32, # num_blocks means how many blocks in a row, note that the padding region of the max/min tensor is not count
    is_causal: cutlass.Constexpr[bool],
    has_lte: cutlass.Constexpr[bool],
    has_uts: cutlass.Constexpr[bool],
    has_ute: cutlass.Constexpr[bool],
    batch_size: cutlass.Int32,
    h_flashmask: cutlass.Int32,
    kBlockM: cutlass.Int32,
    kBlockN: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
):
    # one warp per block row
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bdimx = cute.arch.block_dim()[0]
    warp_per_block = bdimx >> 5
    # make sure num of threads is multiple of 32
    lane_id = tidx & 31
    warp_id = tidx >> 5

    global_warp_id = warp_id + (bidx * bdimx >> 5)
    block_row_idx = global_warp_id % num_blocks_row
    head_idx = (global_warp_id // num_blocks_row) % h_flashmask
    batch_idx = global_warp_id // (h_flashmask * num_blocks_row)

    batch_head_idx = batch_idx * h_flashmask + head_idx

    global_block_row_idx = block_row_idx + head_idx * num_blocks_row + batch_idx * (h_flashmask * num_blocks_row)
    total_num_blocks_row = batch_size * h_flashmask * num_blocks_row

    if global_block_row_idx < total_num_blocks_row:
        row_idx_start = block_row_idx * kBlockM
        row_idx_end = min(row_idx_start + kBlockM, seqlen_q)
        n_block_max = num_blocks_col
        if is_causal:
            # Note(wusiming): make sure window_size_right is 0
            n_idx_right = row_idx_end + seqlen_k - seqlen_q
            n_block_max = min(n_block_max, (n_idx_right + kBlockN - 1) // kBlockN)
        loop_num = (n_block_max + 31) >> 5
        local_sum = 0
        for i in cutlass.range(loop_num):
            block_col_idx = i * 32 + lane_id
            if block_col_idx < n_block_max:
                if cutlass.const_expr(has_uts):
                    if not ((row_idx_start >= LTS_nblock_max[batch_idx, head_idx, block_col_idx] and row_idx_end <= LTE_nblock_min[batch_idx, head_idx, block_col_idx]) or (row_idx_start >= UTS_nblock_max[batch_idx, head_idx, block_col_idx] and row_idx_end <= UTE_nblock_min[batch_idx, head_idx, block_col_idx])):
                        local_sum += 1
                elif cutlass.const_expr(has_lte):
                    if not (row_idx_start >= LTS_nblock_max[batch_idx, head_idx, block_col_idx] and row_idx_end <= LTE_nblock_min[batch_idx, head_idx, block_col_idx]):
                        local_sum += 1
                elif cutlass.const_expr(has_ute):
                    if not (row_idx_start >= LTS_nblock_max[batch_idx, head_idx, block_col_idx] or row_idx_end <= UTE_nblock_min[batch_idx, head_idx, block_col_idx]):
                        local_sum += 1
                else:
                    if not (row_idx_start >= LTS_nblock_max[batch_idx, head_idx, block_col_idx]):
                        local_sum += 1

        warp_sum = utils.warp_reduce(local_sum, operator.add)
        if lane_id == 0:
            valid_block_count[batch_idx, head_idx, block_row_idx] = warp_sum

@cute.jit
def reduce_block_count_cute(
    LTS_nblock_max: cute.Tensor, # [b, h_fm, sk/kBlockN]
    LTE_nblock_min: cute.Tensor,
    UTS_nblock_max: cute.Tensor,
    UTE_nblock_min: cute.Tensor,
    valid_block_count: cute.Tensor, # [b,h_fm, sQ/kBlockM] valid_block_count means how many block are not fully masked in each row
    num_blocks_row: cutlass.Int32,
    num_blocks_col: cutlass.Int32, # num_blocks means how many blocks in a row, note that the padding region of the max/min tensor is not count
    is_causal: cutlass.Constexpr[bool],
    has_lte: cutlass.Constexpr[bool],
    has_uts: cutlass.Constexpr[bool],
    has_ute: cutlass.Constexpr[bool],
    batch_size: cutlass.Int32,
    h_flashmask: cutlass.Int32,
    kBlockM: cutlass.Int32,
    kBlockN: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    stream: cuda.CUstream,
):
    reduce_block_count_kernel(
        LTS_nblock_max if LTS_nblock_max is not None else None,
        LTE_nblock_min if LTE_nblock_min is not None else None,
        UTS_nblock_max if UTS_nblock_max is not None else None,
        UTE_nblock_min if UTE_nblock_min is not None else None,
        valid_block_count,
        num_blocks_row,
        num_blocks_col,
        is_causal,
        has_lte,
        has_uts,
        has_ute,
        batch_size,
        h_flashmask,
        kBlockM,
        kBlockN,
        seqlen_q,
        seqlen_k
    ).launch(
        grid=[(num_blocks_row * batch_size * h_flashmask + 3) >> 2, 1, 1],
        block=[32 * 4, 1, 1],
        stream=stream,
    )

# Note(wusiming): make sure call reduce_block_count after scan_max_min
def reduce_block_count(
    flashmask_info: FlashMaskInfo,
    is_causal: bool,
    kBlockM: int,
    kBlockN: int,
    seqlen_q: int,
):
    batch, heads, seqlen_k, num_vecs = flashmask_info.startend_row_indices.shape
    num_blocks_row = (seqlen_q + kBlockM - 1) // kBlockM
    num_blocks_col = (seqlen_k + kBlockN - 1) // kBlockN
    if num_vecs == 4:
        has_lte = True
        has_uts = True
        has_ute = True
    elif num_vecs == 2:
        if flashmask_info.is_causal:
            has_lte = True
            has_uts = False
            has_ute = False
        else:
            has_lte = False
            has_uts = False
            has_ute = True
    else:
        has_lte = False
        has_uts = False
        has_ute = False

    current_stream = cuda.CUstream(paddle.device.current_stream().stream_base.cuda_stream)

    # TODO(wusiming): Are all of these compile keys necessary?
    compile_key = (is_causal, kBlockM, kBlockN, has_lte, has_uts, has_ute)
    if compile_key not in reduce_block_count.compile_cache:
        reduce_block_count.compile_cache[compile_key] = cute.compile(
            reduce_block_count_cute,
            flashmask_info.LTS_nblock_max,
            flashmask_info.LTE_nblock_min,
            flashmask_info.UTS_nblock_max,
            flashmask_info.UTE_nblock_min,
            flashmask_info.valid_block_count,
            num_blocks_row,
            num_blocks_col,
            is_causal,
            has_lte,
            has_uts,
            has_ute,
            batch,
            heads,
            kBlockM,
            kBlockN,
            seqlen_q,
            seqlen_k,
            current_stream
        )
    reduce_block_count.compile_cache[compile_key](
        flashmask_info.LTS_nblock_max,
        flashmask_info.LTE_nblock_min,
        flashmask_info.UTS_nblock_max,
        flashmask_info.UTE_nblock_min,
        flashmask_info.valid_block_count,
        num_blocks_row,
        num_blocks_col,
        # has_lte,
        # has_uts,
        # has_ute,
        batch,
        heads,
        kBlockM,
        kBlockN,
        seqlen_q,
        seqlen_k,
        current_stream
    )

reduce_block_count.compile_cache = {}
